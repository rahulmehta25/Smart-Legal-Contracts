package queue

import (
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/panjf2000/ants/v2"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// TaskPriority represents task priority levels
type TaskPriority int

const (
	PriorityLow TaskPriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

// String returns string representation of priority
func (tp TaskPriority) String() string {
	switch tp {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// TaskStatus represents the current status of a task
type TaskStatus int

const (
	StatusPending TaskStatus = iota
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusCancelled
	StatusRetrying
)

// String returns string representation of task status
func (ts TaskStatus) String() string {
	switch ts {
	case StatusPending:
		return "pending"
	case StatusRunning:
		return "running"
	case StatusCompleted:
		return "completed"
	case StatusFailed:
		return "failed"
	case StatusCancelled:
		return "cancelled"
	case StatusRetrying:
		return "retrying"
	default:
		return "unknown"
	}
}

// TaskResult represents the result of task execution
type TaskResult struct {
	TaskID    string      `json:"task_id"`
	Status    TaskStatus  `json:"status"`
	Result    interface{} `json:"result,omitempty"`
	Error     string      `json:"error,omitempty"`
	Duration  time.Duration `json:"duration"`
	StartTime time.Time   `json:"start_time"`
	EndTime   time.Time   `json:"end_time"`
	Attempts  int         `json:"attempts"`
}

// TaskFunc represents a function that can be executed as a task
type TaskFunc func(ctx context.Context, args interface{}) (interface{}, error)

// Task represents a unit of work to be executed
type Task struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Priority    TaskPriority  `json:"priority"`
	Args        interface{}   `json:"args"`
	Func        TaskFunc      `json:"-"`
	CreatedAt   time.Time     `json:"created_at"`
	ScheduledAt time.Time     `json:"scheduled_at"`
	Deadline    time.Time     `json:"deadline,omitempty"`
	Timeout     time.Duration `json:"timeout"`
	MaxRetries  int           `json:"max_retries"`
	RetryDelay  time.Duration `json:"retry_delay"`
	Status      TaskStatus    `json:"status"`
	Attempts    int           `json:"attempts"`
	LastError   string        `json:"last_error,omitempty"`
	Tags        []string      `json:"tags,omitempty"`
	
	// Internal fields
	mutex       sync.RWMutex
	ctx         context.Context
	cancel      context.CancelFunc
	resultChan  chan TaskResult
	index       int // For priority queue
}

// NewTask creates a new task
func NewTask(id, name string, priority TaskPriority, fn TaskFunc, args interface{}) *Task {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &Task{
		ID:          id,
		Name:        name,
		Priority:    priority,
		Args:        args,
		Func:        fn,
		CreatedAt:   time.Now(),
		ScheduledAt: time.Now(),
		Timeout:     time.Minute * 5,
		MaxRetries:  3,
		RetryDelay:  time.Second,
		Status:      StatusPending,
		ctx:         ctx,
		cancel:      cancel,
		resultChan:  make(chan TaskResult, 1),
	}
}

// Cancel cancels the task
func (t *Task) Cancel() {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	
	if t.Status == StatusPending || t.Status == StatusRunning {
		t.Status = StatusCancelled
		t.cancel()
	}
}

// GetStatus returns the current task status
func (t *Task) GetStatus() TaskStatus {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	return t.Status
}

// SetStatus updates the task status
func (t *Task) SetStatus(status TaskStatus) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	t.Status = status
}

// IsExpired checks if the task has exceeded its deadline
func (t *Task) IsExpired() bool {
	return !t.Deadline.IsZero() && time.Now().After(t.Deadline)
}

// CanRetry checks if the task can be retried
func (t *Task) CanRetry() bool {
	return t.Attempts < t.MaxRetries
}

// String returns string representation of the task
func (t *Task) String() string {
	return fmt.Sprintf("Task{ID: %s, Name: %s, Priority: %s, Status: %s}", 
		t.ID, t.Name, t.Priority, t.Status)
}

// PriorityQueue implements a priority queue for tasks
type PriorityQueue []*Task

// Len returns the length of the queue
func (pq PriorityQueue) Len() int { return len(pq) }

// Less compares two tasks based on priority and creation time
func (pq PriorityQueue) Less(i, j int) bool {
	// Higher priority tasks come first
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority > pq[j].Priority
	}
	// If same priority, earlier tasks come first
	return pq[i].CreatedAt.Before(pq[j].CreatedAt)
}

// Swap swaps two tasks in the queue
func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

// Push adds a task to the queue
func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	task := x.(*Task)
	task.index = n
	*pq = append(*pq, task)
}

// Pop removes and returns the highest priority task
func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	task := old[n-1]
	old[n-1] = nil
	task.index = -1
	*pq = old[0 : n-1]
	return task
}

// QueueMetrics tracks queue performance
type QueueMetrics struct {
	TotalTasks       int64         `json:"total_tasks"`
	PendingTasks     int64         `json:"pending_tasks"`
	RunningTasks     int64         `json:"running_tasks"`
	CompletedTasks   int64         `json:"completed_tasks"`
	FailedTasks      int64         `json:"failed_tasks"`
	CancelledTasks   int64         `json:"cancelled_tasks"`
	AverageWaitTime  time.Duration `json:"average_wait_time"`
	AverageExecTime  time.Duration `json:"average_exec_time"`
	ThroughputPerSec float64       `json:"throughput_per_sec"`
	ErrorRate        float64       `json:"error_rate"`
	WorkerUtilization float64      `json:"worker_utilization"`
}

// QueueConfig holds configuration for the task queue
type QueueConfig struct {
	MaxWorkers       int           `json:"max_workers"`
	MinWorkers       int           `json:"min_workers"`
	QueueSize        int           `json:"queue_size"`
	WorkerTimeout    time.Duration `json:"worker_timeout"`
	CleanupInterval  time.Duration `json:"cleanup_interval"`
	MetricsInterval  time.Duration `json:"metrics_interval"`
	EnableProfiling  bool          `json:"enable_profiling"`
	RetentionPeriod  time.Duration `json:"retention_period"`
	BatchSize        int           `json:"batch_size"`
	EnableDLQ        bool          `json:"enable_dlq"` // Dead Letter Queue
	MaxConcurrency   int           `json:"max_concurrency"`
}

// DefaultQueueConfig returns a default queue configuration
func DefaultQueueConfig() *QueueConfig {
	return &QueueConfig{
		MaxWorkers:      runtime.NumCPU() * 4,
		MinWorkers:      runtime.NumCPU(),
		QueueSize:       10000,
		WorkerTimeout:   time.Minute * 10,
		CleanupInterval: time.Minute * 5,
		MetricsInterval: time.Second * 30,
		EnableProfiling: true,
		RetentionPeriod: time.Hour * 24,
		BatchSize:       100,
		EnableDLQ:       true,
		MaxConcurrency:  1000,
	}
}

// TaskQueue implements a high-performance task queue with worker pools
type TaskQueue struct {
	config     *QueueConfig
	queue      PriorityQueue
	queueMutex sync.Mutex
	queueCond  *sync.Cond
	
	// Worker pool
	workerPool *ants.Pool
	
	// Task tracking
	tasks        map[string]*Task
	tasksMutex   sync.RWMutex
	
	// Dead letter queue
	dlq          []*Task
	dlqMutex     sync.Mutex
	
	// Metrics
	metrics      *QueueMetrics
	
	// Control
	running      int32
	shutdown     chan struct{}
	workers      sync.WaitGroup
	
	// Rate limiting
	concurrency  *semaphore.Weighted
	
	// Logger
	logger       *zap.Logger
	
	// Callbacks
	onTaskComplete func(*Task, TaskResult)
	onTaskFailed   func(*Task, error)
}

// NewTaskQueue creates a new high-performance task queue
func NewTaskQueue(config *QueueConfig, logger *zap.Logger) (*TaskQueue, error) {
	if config == nil {
		config = DefaultQueueConfig()
	}

	// Create worker pool
	workerPool, err := ants.NewPool(config.MaxWorkers,
		ants.WithOptions(ants.Options{
			ExpiryDuration: config.WorkerTimeout,
			Nonblocking:    true,
			PanicHandler: func(i interface{}) {
				logger.Error("Worker panic", zap.Any("panic", i))
			},
		}))
	if err != nil {
		return nil, fmt.Errorf("failed to create worker pool: %w", err)
	}

	tq := &TaskQueue{
		config:      config,
		queue:       make(PriorityQueue, 0),
		tasks:       make(map[string]*Task),
		dlq:         make([]*Task, 0),
		metrics:     &QueueMetrics{},
		workerPool:  workerPool,
		shutdown:    make(chan struct{}),
		concurrency: semaphore.NewWeighted(int64(config.MaxConcurrency)),
		logger:      logger,
	}

	tq.queueCond = sync.NewCond(&tq.queueMutex)
	heap.Init(&tq.queue)

	return tq, nil
}

// Start starts the task queue
func (tq *TaskQueue) Start() error {
	if !atomic.CompareAndSwapInt32(&tq.running, 0, 1) {
		return fmt.Errorf("task queue is already running")
	}

	tq.logger.Info("Starting task queue", 
		zap.Int("max_workers", tq.config.MaxWorkers),
		zap.Int("queue_size", tq.config.QueueSize))

	// Start background workers
	tq.startBackgroundWorkers()

	// Start dispatcher
	tq.workers.Add(1)
	go tq.dispatcher()

	return nil
}

// Stop gracefully stops the task queue
func (tq *TaskQueue) Stop(timeout time.Duration) error {
	if !atomic.CompareAndSwapInt32(&tq.running, 1, 0) {
		return fmt.Errorf("task queue is not running")
	}

	tq.logger.Info("Stopping task queue")

	// Signal shutdown
	close(tq.shutdown)

	// Wait for workers with timeout
	done := make(chan struct{})
	go func() {
		tq.workers.Wait()
		close(done)
	}()

	select {
	case <-done:
		tq.logger.Info("Task queue stopped gracefully")
	case <-time.After(timeout):
		tq.logger.Warn("Task queue stop timed out")
	}

	// Release worker pool
	tq.workerPool.Release()

	return nil
}

// Submit submits a task to the queue
func (tq *TaskQueue) Submit(task *Task) error {
	if atomic.LoadInt32(&tq.running) == 0 {
		return fmt.Errorf("task queue is not running")
	}

	// Validate task
	if task.Func == nil {
		return fmt.Errorf("task function cannot be nil")
	}

	// Store task
	tq.tasksMutex.Lock()
	tq.tasks[task.ID] = task
	tq.tasksMutex.Unlock()

	// Add to queue
	tq.queueMutex.Lock()
	heap.Push(&tq.queue, task)
	atomic.AddInt64(&tq.metrics.TotalTasks, 1)
	atomic.AddInt64(&tq.metrics.PendingTasks, 1)
	tq.queueCond.Signal()
	tq.queueMutex.Unlock()

	tq.logger.Debug("Task submitted", 
		zap.String("task_id", task.ID),
		zap.String("priority", task.Priority.String()))

	return nil
}

// SubmitBatch submits multiple tasks as a batch
func (tq *TaskQueue) SubmitBatch(tasks []*Task) error {
	if atomic.LoadInt32(&tq.running) == 0 {
		return fmt.Errorf("task queue is not running")
	}

	tq.tasksMutex.Lock()
	tq.queueMutex.Lock()
	defer tq.tasksMutex.Unlock()
	defer tq.queueMutex.Unlock()

	for _, task := range tasks {
		if task.Func == nil {
			continue
		}

		tq.tasks[task.ID] = task
		heap.Push(&tq.queue, task)
		atomic.AddInt64(&tq.metrics.TotalTasks, 1)
		atomic.AddInt64(&tq.metrics.PendingTasks, 1)
	}

	tq.queueCond.Broadcast()

	tq.logger.Debug("Batch submitted", zap.Int("count", len(tasks)))
	return nil
}

// GetTask retrieves a task by ID
func (tq *TaskQueue) GetTask(taskID string) (*Task, bool) {
	tq.tasksMutex.RLock()
	defer tq.tasksMutex.RUnlock()
	task, exists := tq.tasks[taskID]
	return task, exists
}

// CancelTask cancels a task by ID
func (tq *TaskQueue) CancelTask(taskID string) error {
	tq.tasksMutex.RLock()
	task, exists := tq.tasks[taskID]
	tq.tasksMutex.RUnlock()

	if !exists {
		return fmt.Errorf("task not found: %s", taskID)
	}

	task.Cancel()
	atomic.AddInt64(&tq.metrics.CancelledTasks, 1)
	
	tq.logger.Debug("Task cancelled", zap.String("task_id", taskID))
	return nil
}

// GetMetrics returns current queue metrics
func (tq *TaskQueue) GetMetrics() QueueMetrics {
	return QueueMetrics{
		TotalTasks:        atomic.LoadInt64(&tq.metrics.TotalTasks),
		PendingTasks:      atomic.LoadInt64(&tq.metrics.PendingTasks),
		RunningTasks:      atomic.LoadInt64(&tq.metrics.RunningTasks),
		CompletedTasks:    atomic.LoadInt64(&tq.metrics.CompletedTasks),
		FailedTasks:       atomic.LoadInt64(&tq.metrics.FailedTasks),
		CancelledTasks:    atomic.LoadInt64(&tq.metrics.CancelledTasks),
		AverageWaitTime:   tq.metrics.AverageWaitTime,
		AverageExecTime:   tq.metrics.AverageExecTime,
		ThroughputPerSec:  tq.metrics.ThroughputPerSec,
		ErrorRate:         tq.metrics.ErrorRate,
		WorkerUtilization: float64(tq.workerPool.Running()) / float64(tq.workerPool.Cap()),
	}
}

// GetQueueSize returns the current queue size
func (tq *TaskQueue) GetQueueSize() int {
	tq.queueMutex.Lock()
	defer tq.queueMutex.Unlock()
	return tq.queue.Len()
}

// GetDeadLetterQueue returns tasks in the dead letter queue
func (tq *TaskQueue) GetDeadLetterQueue() []*Task {
	tq.dlqMutex.Lock()
	defer tq.dlqMutex.Unlock()
	
	dlq := make([]*Task, len(tq.dlq))
	copy(dlq, tq.dlq)
	return dlq
}

// SetTaskCompleteCallback sets a callback for task completion
func (tq *TaskQueue) SetTaskCompleteCallback(cb func(*Task, TaskResult)) {
	tq.onTaskComplete = cb
}

// SetTaskFailedCallback sets a callback for task failure
func (tq *TaskQueue) SetTaskFailedCallback(cb func(*Task, error)) {
	tq.onTaskFailed = cb
}

// dispatcher dispatches tasks to workers
func (tq *TaskQueue) dispatcher() {
	defer tq.workers.Done()

	for {
		tq.queueMutex.Lock()
		
		// Wait for tasks or shutdown
		for tq.queue.Len() == 0 && atomic.LoadInt32(&tq.running) == 1 {
			tq.queueCond.Wait()
		}

		// Check for shutdown
		if atomic.LoadInt32(&tq.running) == 0 {
			tq.queueMutex.Unlock()
			return
		}

		// Get next task
		task := heap.Pop(&tq.queue).(*Task)
		atomic.AddInt64(&tq.metrics.PendingTasks, -1)
		tq.queueMutex.Unlock()

		// Check if task is expired or cancelled
		if task.IsExpired() || task.GetStatus() == StatusCancelled {
			tq.handleTaskCompletion(task, TaskResult{
				TaskID:   task.ID,
				Status:   StatusCancelled,
				Duration: 0,
			})
			continue
		}

		// Acquire concurrency slot
		if err := tq.concurrency.Acquire(context.Background(), 1); err != nil {
			tq.logger.Error("Failed to acquire concurrency slot", zap.Error(err))
			continue
		}

		// Submit to worker pool
		err := tq.workerPool.Submit(func() {
			defer tq.concurrency.Release(1)
			tq.executeTask(task)
		})

		if err != nil {
			tq.concurrency.Release(1)
			tq.logger.Error("Failed to submit task to worker pool", 
				zap.String("task_id", task.ID), zap.Error(err))
			
			// Retry or move to DLQ
			tq.handleTaskFailure(task, err)
		}
	}
}

// executeTask executes a single task
func (tq *TaskQueue) executeTask(task *Task) {
	startTime := time.Now()
	task.SetStatus(StatusRunning)
	atomic.AddInt64(&tq.metrics.RunningTasks, 1)

	tq.logger.Debug("Executing task", 
		zap.String("task_id", task.ID),
		zap.String("name", task.Name))

	// Create context with timeout
	ctx := task.ctx
	if task.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, task.Timeout)
		defer cancel()
	}

	// Execute task function
	result, err := task.Func(ctx, task.Args)
	endTime := time.Now()
	duration := endTime.Sub(startTime)

	task.Attempts++
	atomic.AddInt64(&tq.metrics.RunningTasks, -1)

	taskResult := TaskResult{
		TaskID:    task.ID,
		StartTime: startTime,
		EndTime:   endTime,
		Duration:  duration,
		Attempts:  task.Attempts,
	}

	if err != nil {
		task.LastError = err.Error()
		taskResult.Status = StatusFailed
		taskResult.Error = err.Error()
		
		// Check if task can be retried
		if task.CanRetry() {
			tq.scheduleRetry(task)
			return
		}

		// Task failed permanently
		tq.handleTaskFailure(task, err)
	} else {
		taskResult.Status = StatusCompleted
		taskResult.Result = result
		task.SetStatus(StatusCompleted)
		atomic.AddInt64(&tq.metrics.CompletedTasks, 1)
	}

	tq.handleTaskCompletion(task, taskResult)
}

// scheduleRetry schedules a task for retry
func (tq *TaskQueue) scheduleRetry(task *Task) {
	task.SetStatus(StatusRetrying)
	
	go func() {
		time.Sleep(task.RetryDelay)
		
		if atomic.LoadInt32(&tq.running) == 1 {
			tq.queueMutex.Lock()
			task.SetStatus(StatusPending)
			heap.Push(&tq.queue, task)
			atomic.AddInt64(&tq.metrics.PendingTasks, 1)
			tq.queueCond.Signal()
			tq.queueMutex.Unlock()
			
			tq.logger.Debug("Task scheduled for retry", 
				zap.String("task_id", task.ID),
				zap.Int("attempt", task.Attempts))
		}
	}()
}

// handleTaskFailure handles permanent task failure
func (tq *TaskQueue) handleTaskFailure(task *Task, err error) {
	task.SetStatus(StatusFailed)
	atomic.AddInt64(&tq.metrics.FailedTasks, 1)

	// Move to dead letter queue if enabled
	if tq.config.EnableDLQ {
		tq.dlqMutex.Lock()
		tq.dlq = append(tq.dlq, task)
		tq.dlqMutex.Unlock()
	}

	// Call failure callback
	if tq.onTaskFailed != nil {
		go tq.onTaskFailed(task, err)
	}

	tq.logger.Error("Task failed permanently", 
		zap.String("task_id", task.ID),
		zap.String("name", task.Name),
		zap.Error(err))
}

// handleTaskCompletion handles task completion
func (tq *TaskQueue) handleTaskCompletion(task *Task, result TaskResult) {
	// Send result to result channel
	select {
	case task.resultChan <- result:
	default:
		// Channel is full or closed
	}

	// Call completion callback
	if tq.onTaskComplete != nil {
		go tq.onTaskComplete(task, result)
	}
}

// startBackgroundWorkers starts background maintenance tasks
func (tq *TaskQueue) startBackgroundWorkers() {
	// Metrics collector
	tq.workers.Add(1)
	go func() {
		defer tq.workers.Done()
		ticker := time.NewTicker(tq.config.MetricsInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				tq.updateMetrics()
			case <-tq.shutdown:
				return
			}
		}
	}()

	// Cleanup worker
	tq.workers.Add(1)
	go func() {
		defer tq.workers.Done()
		ticker := time.NewTicker(tq.config.CleanupInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				tq.cleanup()
			case <-tq.shutdown:
				return
			}
		}
	}()
}

// updateMetrics calculates and updates queue metrics
func (tq *TaskQueue) updateMetrics() {
	// Calculate throughput
	completed := atomic.LoadInt64(&tq.metrics.CompletedTasks)
	failed := atomic.LoadInt64(&tq.metrics.FailedTasks)
	total := completed + failed

	if total > 0 {
		tq.metrics.ErrorRate = float64(failed) / float64(total)
	}

	// Calculate worker utilization
	tq.metrics.WorkerUtilization = float64(tq.workerPool.Running()) / float64(tq.workerPool.Cap())

	tq.logger.Debug("Metrics updated",
		zap.Float64("error_rate", tq.metrics.ErrorRate),
		zap.Float64("worker_utilization", tq.metrics.WorkerUtilization))
}

// cleanup removes old completed tasks
func (tq *TaskQueue) cleanup() {
	cutoff := time.Now().Add(-tq.config.RetentionPeriod)
	
	tq.tasksMutex.Lock()
	toRemove := make([]string, 0)
	
	for id, task := range tq.tasks {
		if task.GetStatus() == StatusCompleted || task.GetStatus() == StatusFailed {
			if task.CreatedAt.Before(cutoff) {
				toRemove = append(toRemove, id)
			}
		}
	}
	
	for _, id := range toRemove {
		delete(tq.tasks, id)
	}
	tq.tasksMutex.Unlock()

	if len(toRemove) > 0 {
		tq.logger.Debug("Cleaned up old tasks", zap.Int("count", len(toRemove)))
	}
}