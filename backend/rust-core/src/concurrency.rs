//! Concurrency system with Tokio and Actix

use crate::{CoreError, CoreResult};
use actix::{Actor, ActorContext, AsyncContext, Context, Handler, Message, Recipient, StreamHandler};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock as AsyncRwLock, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, timeout};

/// Concurrency configuration
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    pub max_workers: usize,
    pub max_queue_size: usize,
    pub task_timeout: Duration,
    pub enable_backpressure: bool,
    pub worker_idle_timeout: Duration,
    pub enable_metrics: bool,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get() * 2,
            max_queue_size: 10000,
            task_timeout: Duration::from_secs(30),
            enable_backpressure: true,
            worker_idle_timeout: Duration::from_secs(60),
            enable_metrics: true,
        }
    }
}

/// Work-stealing task scheduler
pub struct WorkStealingScheduler {
    workers: Vec<WorkerHandle>,
    global_queue: Arc<crossbeam_deque::Injector<Task>>,
    config: ConcurrencyConfig,
    metrics: Arc<SchedulerMetrics>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

/// Worker handle for work-stealing
struct WorkerHandle {
    id: usize,
    local_queue: Arc<crossbeam_deque::Worker<Task>>,
    stealer: crossbeam_deque::Stealer<Task>,
    join_handle: JoinHandle<()>,
}

/// Generic task for the scheduler
#[derive(Debug)]
pub struct Task {
    id: u64,
    name: String,
    payload: TaskPayload,
    priority: TaskPriority,
    created_at: Instant,
    deadline: Option<Instant>,
}

#[derive(Debug)]
pub enum TaskPayload {
    Computation(Box<dyn FnOnce() -> TaskResult + Send + 'static>),
    AsyncComputation(Box<dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult> + Send>> + Send + 'static>),
    Message(Box<dyn std::any::Any + Send>),
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
    pub execution_time: Duration,
}

/// Scheduler metrics
#[derive(Debug, Default)]
struct SchedulerMetrics {
    tasks_submitted: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    total_execution_time: AtomicU64,
    active_workers: AtomicUsize,
    queue_size: AtomicUsize,
}

/// Actor-based message processing system
pub struct MessageBus {
    actors: HashMap<String, Box<dyn ActorHandle>>,
    message_router: MessageRouter,
    config: ConcurrencyConfig,
}

trait ActorHandle: Send {
    fn send_message(&self, message: Box<dyn std::any::Any + Send>) -> CoreResult<()>;
    fn stop(&self) -> CoreResult<()>;
}

/// Message router for actor system
#[derive(Clone)]
pub struct MessageRouter {
    routes: Arc<AsyncRwLock<HashMap<String, Vec<String>>>>,
    subscribers: Arc<AsyncRwLock<HashMap<String, Vec<mpsc::UnboundedSender<RouterMessage>>>>>,
}

#[derive(Debug, Clone)]
pub struct RouterMessage {
    pub topic: String,
    pub payload: serde_json::Value,
    pub sender: Option<String>,
    pub timestamp: Instant,
}

/// Async task pool with backpressure
pub struct AsyncTaskPool {
    semaphore: Arc<Semaphore>,
    active_tasks: Arc<AtomicUsize>,
    completed_tasks: Arc<AtomicU64>,
    failed_tasks: Arc<AtomicU64>,
    config: ConcurrencyConfig,
}

/// Channel-based communication system
pub struct ChannelSystem {
    channels: HashMap<String, ChannelHandle>,
    config: ConcurrencyConfig,
}

enum ChannelHandle {
    Bounded(Sender<ChannelMessage>, Receiver<ChannelMessage>),
    Unbounded(crossbeam_channel::Sender<ChannelMessage>, crossbeam_channel::Receiver<ChannelMessage>),
    Async(mpsc::UnboundedSender<ChannelMessage>, Arc<AsyncRwLock<mpsc::UnboundedReceiver<ChannelMessage>>>),
}

#[derive(Debug, Clone)]
pub struct ChannelMessage {
    pub id: String,
    pub data: serde_json::Value,
    pub reply_to: Option<oneshot::Sender<ChannelMessage>>,
    pub timestamp: Instant,
}

/// Process manager for long-running tasks
pub struct ProcessManager {
    processes: Arc<AsyncRwLock<HashMap<String, ProcessHandle>>>,
    config: ConcurrencyConfig,
    cleanup_interval: Option<JoinHandle<()>>,
}

#[derive(Debug)]
struct ProcessHandle {
    id: String,
    name: String,
    join_handle: JoinHandle<ProcessResult>,
    started_at: Instant,
    status: ProcessStatus,
    kill_switch: oneshot::Sender<()>,
}

#[derive(Debug, Clone)]
pub enum ProcessStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub id: String,
    pub success: bool,
    pub runtime: Duration,
    pub error: Option<String>,
}

impl WorkStealingScheduler {
    /// Create new work-stealing scheduler
    pub fn new(config: ConcurrencyConfig) -> CoreResult<Self> {
        let global_queue = Arc::new(crossbeam_deque::Injector::new());
        let metrics = Arc::new(SchedulerMetrics::default());
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
        
        let mut workers = Vec::new();
        let mut stealers = Vec::new();

        // Create workers
        for worker_id in 0..config.max_workers {
            let local_queue = Arc::new(crossbeam_deque::Worker::new_fifo());
            let stealer = local_queue.stealer();
            stealers.push(stealer.clone());

            let worker_handle = WorkerHandle {
                id: worker_id,
                local_queue: local_queue.clone(),
                stealer,
                join_handle: tokio::spawn(Self::worker_loop(
                    worker_id,
                    local_queue,
                    global_queue.clone(),
                    stealers.clone(),
                    metrics.clone(),
                    config.clone(),
                    shutdown_rx.resubscribe(),
                )),
            };

            workers.push(worker_handle);
        }

        Ok(Self {
            workers,
            global_queue,
            config,
            metrics,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    /// Submit task to scheduler
    pub fn submit_task<F>(&self, name: String, priority: TaskPriority, f: F) -> CoreResult<u64>
    where
        F: FnOnce() -> TaskResult + Send + 'static,
    {
        if self.metrics.queue_size.load(Ordering::Relaxed) >= self.config.max_queue_size {
            if self.config.enable_backpressure {
                return Err(CoreError::CacheError("Queue is full".to_string()));
            }
        }

        let task_id = self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        let task = Task {
            id: task_id,
            name,
            payload: TaskPayload::Computation(Box::new(f)),
            priority,
            created_at: Instant::now(),
            deadline: None,
        };

        self.global_queue.push(task);
        self.metrics.queue_size.fetch_add(1, Ordering::Relaxed);

        Ok(task_id)
    }

    /// Submit async task to scheduler
    pub fn submit_async_task<F, Fut>(&self, name: String, priority: TaskPriority, f: F) -> CoreResult<u64>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = TaskResult> + Send + 'static,
    {
        let task_id = self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        let task = Task {
            id: task_id,
            name,
            payload: TaskPayload::AsyncComputation(Box::new(move || Box::pin(f()))),
            priority,
            created_at: Instant::now(),
            deadline: None,
        };

        self.global_queue.push(task);
        self.metrics.queue_size.fetch_add(1, Ordering::Relaxed);

        Ok(task_id)
    }

    /// Worker loop implementation
    async fn worker_loop(
        worker_id: usize,
        local_queue: Arc<crossbeam_deque::Worker<Task>>,
        global_queue: Arc<crossbeam_deque::Injector<Task>>,
        stealers: Vec<crossbeam_deque::Stealer<Task>>,
        metrics: Arc<SchedulerMetrics>,
        config: ConcurrencyConfig,
        mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    ) {
        metrics.active_workers.fetch_add(1, Ordering::Relaxed);
        
        let mut idle_timer = interval(config.worker_idle_timeout);
        
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::debug!("Worker {} shutting down", worker_id);
                    break;
                }
                _ = idle_timer.tick() => {
                    // Worker idle timeout - could implement worker scaling here
                }
                _ = tokio::task::yield_now() => {
                    if let Some(task) = Self::find_task(&local_queue, &global_queue, &stealers) {
                        Self::execute_task(task, &metrics, &config).await;
                    }
                }
            }
        }
        
        metrics.active_workers.fetch_sub(1, Ordering::Relaxed);
    }

    /// Find task using work-stealing algorithm
    fn find_task(
        local_queue: &crossbeam_deque::Worker<Task>,
        global_queue: &crossbeam_deque::Injector<Task>,
        stealers: &[crossbeam_deque::Stealer<Task>],
    ) -> Option<Task> {
        // Try local queue first
        if let Some(task) = local_queue.pop() {
            return Some(task);
        }

        // Try global queue
        if let Some(task) = global_queue.steal().success() {
            return Some(task);
        }

        // Try stealing from other workers
        for stealer in stealers {
            if let Some(task) = stealer.steal().success() {
                return Some(task);
            }
        }

        None
    }

    /// Execute task
    async fn execute_task(task: Task, metrics: &Arc<SchedulerMetrics>, config: &ConcurrencyConfig) {
        let start_time = Instant::now();
        let task_timeout = config.task_timeout;
        
        metrics.queue_size.fetch_sub(1, Ordering::Relaxed);

        let result = match task.payload {
            TaskPayload::Computation(f) => {
                // Run CPU-bound task in blocking thread
                let result = tokio::task::spawn_blocking(move || f()).await;
                match result {
                    Ok(task_result) => task_result,
                    Err(e) => TaskResult {
                        success: false,
                        data: None,
                        error: Some(e.to_string()),
                        execution_time: start_time.elapsed(),
                    },
                }
            }
            TaskPayload::AsyncComputation(f) => {
                let future = f();
                match timeout(task_timeout, future).await {
                    Ok(result) => result,
                    Err(_) => TaskResult {
                        success: false,
                        data: None,
                        error: Some("Task timeout".to_string()),
                        execution_time: start_time.elapsed(),
                    },
                }
            }
            TaskPayload::Message(_) => {
                TaskResult {
                    success: true,
                    data: None,
                    error: None,
                    execution_time: start_time.elapsed(),
                }
            }
        };

        // Update metrics
        if result.success {
            metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
        } else {
            metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
        }

        metrics.total_execution_time.fetch_add(
            result.execution_time.as_nanos() as u64,
            Ordering::Relaxed,
        );

        tracing::debug!(
            "Task {} completed: success={}, duration={:?}",
            task.id,
            result.success,
            result.execution_time
        );
    }

    /// Get scheduler metrics
    pub fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            tasks_submitted: AtomicU64::new(self.metrics.tasks_submitted.load(Ordering::Relaxed)),
            tasks_completed: AtomicU64::new(self.metrics.tasks_completed.load(Ordering::Relaxed)),
            tasks_failed: AtomicU64::new(self.metrics.tasks_failed.load(Ordering::Relaxed)),
            total_execution_time: AtomicU64::new(self.metrics.total_execution_time.load(Ordering::Relaxed)),
            active_workers: AtomicUsize::new(self.metrics.active_workers.load(Ordering::Relaxed)),
            queue_size: AtomicUsize::new(self.metrics.queue_size.load(Ordering::Relaxed)),
        }
    }

    /// Shutdown scheduler gracefully
    pub async fn shutdown(mut self) -> CoreResult<()> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(()).await;
        }

        // Wait for all workers to complete
        for worker in self.workers {
            if let Err(e) = worker.join_handle.await {
                tracing::error!("Worker {} failed to shutdown: {}", worker.id, e);
            }
        }

        Ok(())
    }
}

impl MessageRouter {
    /// Create new message router
    pub fn new() -> Self {
        Self {
            routes: Arc::new(AsyncRwLock::new(HashMap::new())),
            subscribers: Arc::new(AsyncRwLock::new(HashMap::new())),
        }
    }

    /// Add route from topic to destination
    pub async fn add_route(&self, from_topic: String, to_topics: Vec<String>) {
        let mut routes = self.routes.write().await;
        routes.insert(from_topic, to_topics);
    }

    /// Subscribe to topic
    pub async fn subscribe(&self, topic: String) -> mpsc::UnboundedReceiver<RouterMessage> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let mut subscribers = self.subscribers.write().await;
        subscribers.entry(topic).or_insert_with(Vec::new).push(tx);
        
        rx
    }

    /// Publish message to topic
    pub async fn publish(&self, message: RouterMessage) -> CoreResult<()> {
        let subscribers = self.subscribers.read().await;
        
        if let Some(topic_subscribers) = subscribers.get(&message.topic) {
            for subscriber in topic_subscribers {
                if subscriber.send(message.clone()).is_err() {
                    tracing::warn!("Failed to send message to subscriber for topic: {}", message.topic);
                }
            }
        }

        // Forward to routed topics
        let routes = self.routes.read().await;
        if let Some(destinations) = routes.get(&message.topic) {
            for destination in destinations {
                let mut forwarded_message = message.clone();
                forwarded_message.topic = destination.clone();
                
                if let Some(dest_subscribers) = subscribers.get(destination) {
                    for subscriber in dest_subscribers {
                        if subscriber.send(forwarded_message.clone()).is_err() {
                            tracing::warn!("Failed to forward message to topic: {}", destination);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl AsyncTaskPool {
    /// Create new async task pool
    pub fn new(config: ConcurrencyConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_workers)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
            completed_tasks: Arc::new(AtomicU64::new(0)),
            failed_tasks: Arc::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Spawn task with backpressure control
    pub async fn spawn<F, T>(&self, future: F) -> CoreResult<JoinHandle<T>>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.semaphore.acquire().await
            .map_err(|e| CoreError::CacheError(e.to_string()))?;
        
        let active_tasks = self.active_tasks.clone();
        let completed_tasks = self.completed_tasks.clone();
        let failed_tasks = self.failed_tasks.clone();
        
        active_tasks.fetch_add(1, Ordering::Relaxed);
        
        let handle = tokio::spawn(async move {
            let _permit = permit; // Keep permit alive
            let result = future.await;
            
            active_tasks.fetch_sub(1, Ordering::Relaxed);
            completed_tasks.fetch_add(1, Ordering::Relaxed);
            
            result
        });

        Ok(handle)
    }

    /// Spawn with timeout
    pub async fn spawn_with_timeout<F, T>(
        &self,
        future: F,
        timeout_duration: Duration,
    ) -> CoreResult<JoinHandle<Result<T, CoreError>>>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let task_future = async move {
            match timeout(timeout_duration, future).await {
                Ok(result) => Ok(result),
                Err(_) => Err(CoreError::CacheError("Task timeout".to_string())),
            }
        };

        self.spawn(task_future).await
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, u64, u64) {
        (
            self.active_tasks.load(Ordering::Relaxed),
            self.completed_tasks.load(Ordering::Relaxed),
            self.failed_tasks.load(Ordering::Relaxed),
        )
    }
}

impl ProcessManager {
    /// Create new process manager
    pub fn new(config: ConcurrencyConfig) -> Self {
        let manager = Self {
            processes: Arc::new(AsyncRwLock::new(HashMap::new())),
            config,
            cleanup_interval: None,
        };

        // Start cleanup task
        let processes = manager.processes.clone();
        let cleanup_handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                Self::cleanup_finished_processes(&processes).await;
            }
        });

        Self {
            cleanup_interval: Some(cleanup_handle),
            ..manager
        }
    }

    /// Start long-running process
    pub async fn start_process<F, Fut>(&self, id: String, name: String, f: F) -> CoreResult<()>
    where
        F: FnOnce(oneshot::Receiver<()>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ProcessResult> + Send + 'static,
    {
        let (kill_tx, kill_rx) = oneshot::channel();
        
        let join_handle = tokio::spawn(f(kill_rx));
        
        let process = ProcessHandle {
            id: id.clone(),
            name,
            join_handle,
            started_at: Instant::now(),
            status: ProcessStatus::Starting,
            kill_switch: kill_tx,
        };

        let mut processes = self.processes.write().await;
        processes.insert(id, process);

        Ok(())
    }

    /// Stop process
    pub async fn stop_process(&self, id: &str) -> CoreResult<()> {
        let mut processes = self.processes.write().await;
        
        if let Some(mut process) = processes.remove(id) {
            process.status = ProcessStatus::Stopping;
            
            // Send kill signal
            let _ = process.kill_switch.send(());
            
            // Wait for process to finish
            if let Ok(result) = process.join_handle.await {
                tracing::info!("Process {} stopped: {:?}", id, result);
            }
        }

        Ok(())
    }

    /// List all processes
    pub async fn list_processes(&self) -> HashMap<String, (ProcessStatus, Duration)> {
        let processes = self.processes.read().await;
        
        processes
            .iter()
            .map(|(id, process)| {
                (
                    id.clone(),
                    (process.status.clone(), process.started_at.elapsed()),
                )
            })
            .collect()
    }

    /// Cleanup finished processes
    async fn cleanup_finished_processes(processes: &AsyncRwLock<HashMap<String, ProcessHandle>>) {
        let mut to_remove = Vec::new();
        
        {
            let processes_read = processes.read().await;
            for (id, process) in processes_read.iter() {
                if process.join_handle.is_finished() {
                    to_remove.push(id.clone());
                }
            }
        }
        
        if !to_remove.is_empty() {
            let mut processes_write = processes.write().await;
            for id in to_remove {
                if let Some(process) = processes_write.remove(&id) {
                    tracing::debug!("Cleaned up finished process: {}", process.name);
                }
            }
        }
    }
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        if let Some(cleanup_handle) = self.cleanup_interval.take() {
            cleanup_handle.abort();
        }
    }
}

/// Utility functions for concurrency patterns
pub mod patterns {
    use super::*;

    /// Fan-out pattern: distribute work to multiple workers
    pub async fn fan_out<T, F, Fut>(
        items: Vec<T>,
        worker_count: usize,
        worker_fn: F,
    ) -> Vec<Result<Fut::Output, CoreError>>
    where
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let semaphore = Arc::new(Semaphore::new(worker_count));
        let mut handles = Vec::new();

        for item in items {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let worker_fn = worker_fn.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                worker_fn(item).await
            });
            
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(Ok(result)),
                Err(e) => results.push(Err(CoreError::CacheError(e.to_string()))),
            }
        }

        results
    }

    /// Pipeline pattern: process items through stages
    pub async fn pipeline<T>(
        mut input: Vec<T>,
        stages: Vec<Box<dyn Fn(T) -> CoreResult<T> + Send + Sync>>,
    ) -> CoreResult<Vec<T>>
    where
        T: Send + 'static,
    {
        for stage in stages {
            let mut stage_results = Vec::new();
            
            for item in input {
                match stage(item) {
                    Ok(result) => stage_results.push(result),
                    Err(e) => return Err(e),
                }
            }
            
            input = stage_results;
        }

        Ok(input)
    }

    /// Circuit breaker pattern
    pub struct CircuitBreaker {
        failure_count: AtomicUsize,
        last_failure_time: Arc<AsyncRwLock<Option<Instant>>>,
        failure_threshold: usize,
        timeout: Duration,
        state: Arc<AsyncRwLock<CircuitState>>,
    }

    #[derive(Debug, Clone)]
    pub enum CircuitState {
        Closed,
        Open,
        HalfOpen,
    }

    impl CircuitBreaker {
        pub fn new(failure_threshold: usize, timeout: Duration) -> Self {
            Self {
                failure_count: AtomicUsize::new(0),
                last_failure_time: Arc::new(AsyncRwLock::new(None)),
                failure_threshold,
                timeout,
                state: Arc::new(AsyncRwLock::new(CircuitState::Closed)),
            }
        }

        pub async fn call<F, T>(&self, f: F) -> CoreResult<T>
        where
            F: std::future::Future<Output = CoreResult<T>>,
        {
            let state = self.state.read().await.clone();
            
            match state {
                CircuitState::Open => {
                    let last_failure = self.last_failure_time.read().await;
                    if let Some(last_failure) = *last_failure {
                        if last_failure.elapsed() > self.timeout {
                            *self.state.write().await = CircuitState::HalfOpen;
                        } else {
                            return Err(CoreError::CacheError("Circuit breaker is open".to_string()));
                        }
                    }
                }
                _ => {}
            }

            match f.await {
                Ok(result) => {
                    self.on_success().await;
                    Ok(result)
                }
                Err(e) => {
                    self.on_failure().await;
                    Err(e)
                }
            }
        }

        async fn on_success(&self) {
            self.failure_count.store(0, Ordering::Relaxed);
            *self.state.write().await = CircuitState::Closed;
        }

        async fn on_failure(&self) {
            let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
            *self.last_failure_time.write().await = Some(Instant::now());
            
            if count >= self.failure_threshold {
                *self.state.write().await = CircuitState::Open;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new(ConcurrencyConfig::default()).unwrap();
        
        let task_id = scheduler.submit_task(
            "test_task".to_string(),
            TaskPriority::Normal,
            || TaskResult {
                success: true,
                data: Some(serde_json::json!({"result": "test"})),
                error: None,
                execution_time: Duration::from_millis(10),
            },
        ).unwrap();
        
        assert!(task_id > 0);
        
        // Wait a bit for task to complete
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let metrics = scheduler.metrics();
        assert!(metrics.tasks_submitted.load(Ordering::Relaxed) > 0);
        
        scheduler.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_message_router() {
        let router = MessageRouter::new();
        
        router.add_route("input".to_string(), vec!["output".to_string()]).await;
        let mut rx = router.subscribe("output".to_string()).await;
        
        let message = RouterMessage {
            topic: "input".to_string(),
            payload: serde_json::json!({"data": "test"}),
            sender: None,
            timestamp: Instant::now(),
        };
        
        router.publish(message).await.unwrap();
        
        let received = rx.recv().await.unwrap();
        assert_eq!(received.topic, "output");
    }

    #[tokio::test]
    async fn test_async_task_pool() {
        let pool = AsyncTaskPool::new(ConcurrencyConfig {
            max_workers: 2,
            ..Default::default()
        });
        
        let handle = pool.spawn(async { "test result" }).await.unwrap();
        let result = handle.await.unwrap();
        
        assert_eq!(result, "test result");
        
        let (active, completed, _failed) = pool.stats();
        assert_eq!(active, 0);
        assert_eq!(completed, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = patterns::CircuitBreaker::new(2, Duration::from_millis(100));
        
        // Successful call
        let result = breaker.call(async { Ok::<_, CoreError>("success") }).await;
        assert!(result.is_ok());
        
        // Failed calls to trip the breaker
        let _ = breaker.call(async { Err::<String, _>(CoreError::CacheError("fail".to_string())) }).await;
        let _ = breaker.call(async { Err::<String, _>(CoreError::CacheError("fail".to_string())) }).await;
        
        // Should be open now
        let result = breaker.call(async { Ok::<_, CoreError>("should fail") }).await;
        assert!(result.is_err());
    }
}