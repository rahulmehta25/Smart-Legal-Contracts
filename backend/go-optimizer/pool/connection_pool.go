package pool

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// ConnectionState represents the state of a connection
type ConnectionState int

const (
	StateIdle ConnectionState = iota
	StateActive
	StateUnhealthy
	StateClosed
)

// String returns string representation of connection state
func (cs ConnectionState) String() string {
	switch cs {
	case StateIdle:
		return "idle"
	case StateActive:
		return "active"
	case StateUnhealthy:
		return "unhealthy"
	case StateClosed:
		return "closed"
	default:
		return "unknown"
	}
}

// Connection wraps a network connection with metadata
type Connection struct {
	conn        net.Conn
	id          string
	createdAt   time.Time
	lastUsed    time.Time
	usageCount  int64
	state       ConnectionState
	healthScore float64
	mutex       sync.RWMutex
	pool        *ConnectionPool
}

// NewConnection creates a new connection wrapper
func NewConnection(conn net.Conn, id string, pool *ConnectionPool) *Connection {
	return &Connection{
		conn:        conn,
		id:          id,
		createdAt:   time.Now(),
		lastUsed:    time.Now(),
		state:       StateIdle,
		healthScore: 1.0,
		pool:        pool,
	}
}

// ID returns the connection ID
func (c *Connection) ID() string {
	return c.id
}

// State returns the current connection state
func (c *Connection) State() ConnectionState {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.state
}

// SetState updates the connection state
func (c *Connection) SetState(state ConnectionState) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.state = state
}

// HealthScore returns the current health score (0.0 to 1.0)
func (c *Connection) HealthScore() float64 {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.healthScore
}

// UpdateHealthScore updates the health score
func (c *Connection) UpdateHealthScore(score float64) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.healthScore = score
}

// IsHealthy returns true if the connection is healthy
func (c *Connection) IsHealthy() bool {
	return c.HealthScore() > 0.5 && c.State() != StateUnhealthy && c.State() != StateClosed
}

// Use marks the connection as being used
func (c *Connection) Use() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.lastUsed = time.Now()
	atomic.AddInt64(&c.usageCount, 1)
	c.state = StateActive
}

// Release marks the connection as idle
func (c *Connection) Release() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.lastUsed = time.Now()
	c.state = StateIdle
}

// Age returns the connection age
func (c *Connection) Age() time.Duration {
	return time.Since(c.createdAt)
}

// IdleTime returns how long the connection has been idle
func (c *Connection) IdleTime() time.Duration {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	if c.state == StateActive {
		return 0
	}
	return time.Since(c.lastUsed)
}

// UsageCount returns the number of times this connection has been used
func (c *Connection) UsageCount() int64 {
	return atomic.LoadInt64(&c.usageCount)
}

// Close closes the underlying connection
func (c *Connection) Close() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.state = StateClosed
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Read implements io.Reader
func (c *Connection) Read(b []byte) (n int, err error) {
	return c.conn.Read(b)
}

// Write implements io.Writer
func (c *Connection) Write(b []byte) (n int, err error) {
	return c.conn.Write(b)
}

// SetDeadline sets read and write deadlines
func (c *Connection) SetDeadline(t time.Time) error {
	return c.conn.SetDeadline(t)
}

// SetReadDeadline sets read deadline
func (c *Connection) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

// SetWriteDeadline sets write deadline
func (c *Connection) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}

// RemoteAddr returns the remote address
func (c *Connection) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

// LocalAddr returns the local address
func (c *Connection) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

// PoolConfig holds configuration for the connection pool
type PoolConfig struct {
	MaxConnections      int           // Maximum number of connections
	MinConnections      int           // Minimum number of connections to maintain
	MaxIdleTime         time.Duration // Maximum idle time before closing
	MaxConnectionAge    time.Duration // Maximum age before forcing renewal
	ConnectionTimeout   time.Duration // Timeout for establishing new connections
	HealthCheckInterval time.Duration // Interval for health checks
	RetryAttempts       int           // Number of retry attempts for failed connections
	RetryDelay          time.Duration // Delay between retry attempts
	EnableTLS           bool          // Enable TLS connections
	TLSConfig           *tls.Config   // TLS configuration
	KeepAlive           time.Duration // TCP keep-alive interval
	NoDelay             bool          // TCP_NODELAY option
}

// DefaultPoolConfig returns a default pool configuration
func DefaultPoolConfig() *PoolConfig {
	return &PoolConfig{
		MaxConnections:      100,
		MinConnections:      5,
		MaxIdleTime:         time.Minute * 30,
		MaxConnectionAge:    time.Hour * 24,
		ConnectionTimeout:   time.Second * 30,
		HealthCheckInterval: time.Minute * 5,
		RetryAttempts:       3,
		RetryDelay:          time.Second,
		KeepAlive:           time.Minute * 15,
		NoDelay:             true,
	}
}

// PoolMetrics tracks pool performance
type PoolMetrics struct {
	TotalConnections    int64
	ActiveConnections   int64
	IdleConnections     int64
	UnhealthyConnections int64
	ConnectionsCreated  int64
	ConnectionsDestroyed int64
	ConnectionErrors    int64
	HealthChecksPassed  int64
	HealthChecksFailed  int64
	AverageResponseTime time.Duration
}

// CircuitBreakerState represents circuit breaker states
type CircuitBreakerState int

const (
	CircuitClosed CircuitBreakerState = iota
	CircuitOpen
	CircuitHalfOpen
)

// CircuitBreaker implements circuit breaker pattern for connection pool
type CircuitBreaker struct {
	maxFailures int
	timeout     time.Duration
	state       CircuitBreakerState
	failures    int64
	lastFailure time.Time
	mutex       sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures: maxFailures,
		timeout:     timeout,
		state:       CircuitClosed,
	}
}

// CanExecute returns true if the circuit allows execution
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		return time.Since(cb.lastFailure) > cb.timeout
	case CircuitHalfOpen:
		return true
	default:
		return false
	}
}

// OnSuccess records a successful execution
func (cb *CircuitBreaker) OnSuccess() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	cb.failures = 0
	cb.state = CircuitClosed
}

// OnFailure records a failed execution
func (cb *CircuitBreaker) OnFailure() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	cb.failures++
	cb.lastFailure = time.Now()
	
	if cb.failures >= int64(cb.maxFailures) {
		cb.state = CircuitOpen
	}
}

// State returns the current circuit breaker state
func (cb *CircuitBreaker) State() CircuitBreakerState {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.state
}

// ConnectionPool manages a pool of network connections with health monitoring
type ConnectionPool struct {
	config          *PoolConfig
	connections     map[string]*Connection
	idleConnections chan *Connection
	activeConns     map[string]*Connection
	semaphore       *semaphore.Weighted
	circuitBreaker  *CircuitBreaker
	metrics         *PoolMetrics
	logger          *zap.Logger
	mutex           sync.RWMutex
	shutdown        chan struct{}
	workers         sync.WaitGroup
	dialFunc        func() (net.Conn, error)
	healthCheckFunc func(*Connection) bool
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(config *PoolConfig, dialFunc func() (net.Conn, error), logger *zap.Logger) *ConnectionPool {
	if config == nil {
		config = DefaultPoolConfig()
	}

	pool := &ConnectionPool{
		config:          config,
		connections:     make(map[string]*Connection),
		idleConnections: make(chan *Connection, config.MaxConnections),
		activeConns:     make(map[string]*Connection),
		semaphore:       semaphore.NewWeighted(int64(config.MaxConnections)),
		circuitBreaker:  NewCircuitBreaker(10, time.Minute*5),
		metrics:         &PoolMetrics{},
		logger:          logger,
		shutdown:        make(chan struct{}),
		dialFunc:        dialFunc,
		healthCheckFunc: defaultHealthCheck,
	}

	// Start background workers
	pool.startBackgroundWorkers()

	// Pre-populate with minimum connections
	pool.warmUp()

	return pool
}

// Get acquires a connection from the pool
func (cp *ConnectionPool) Get(ctx context.Context) (*Connection, error) {
	// Check circuit breaker
	if !cp.circuitBreaker.CanExecute() {
		return nil, fmt.Errorf("circuit breaker is open")
	}

	// Acquire semaphore
	if err := cp.semaphore.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("failed to acquire semaphore: %w", err)
	}

	// Try to get an idle connection first
	select {
	case conn := <-cp.idleConnections:
		if conn.IsHealthy() {
			conn.Use()
			cp.moveToActive(conn)
			cp.circuitBreaker.OnSuccess()
			atomic.AddInt64(&cp.metrics.ActiveConnections, 1)
			atomic.AddInt64(&cp.metrics.IdleConnections, -1)
			return conn, nil
		}
		// Connection is unhealthy, close it and create a new one
		conn.Close()
		cp.removeConnection(conn.ID())
	default:
		// No idle connections available
	}

	// Create a new connection
	conn, err := cp.createConnection()
	if err != nil {
		cp.semaphore.Release(1)
		cp.circuitBreaker.OnFailure()
		atomic.AddInt64(&cp.metrics.ConnectionErrors, 1)
		return nil, fmt.Errorf("failed to create connection: %w", err)
	}

	conn.Use()
	cp.moveToActive(conn)
	cp.circuitBreaker.OnSuccess()
	atomic.AddInt64(&cp.metrics.ActiveConnections, 1)
	atomic.AddInt64(&cp.metrics.ConnectionsCreated, 1)

	return conn, nil
}

// Put returns a connection to the pool
func (cp *ConnectionPool) Put(conn *Connection) error {
	if conn == nil {
		return fmt.Errorf("connection is nil")
	}

	defer cp.semaphore.Release(1)

	cp.mutex.Lock()
	defer cp.mutex.Unlock()

	// Remove from active connections
	delete(cp.activeConns, conn.ID())
	atomic.AddInt64(&cp.metrics.ActiveConnections, -1)

	// Check if connection is still healthy
	if !conn.IsHealthy() || conn.Age() > cp.config.MaxConnectionAge {
		conn.Close()
		cp.removeConnectionUnsafe(conn.ID())
		atomic.AddInt64(&cp.metrics.ConnectionsDestroyed, 1)
		return nil
	}

	// Return to idle pool
	conn.Release()
	
	select {
	case cp.idleConnections <- conn:
		atomic.AddInt64(&cp.metrics.IdleConnections, 1)
		return nil
	default:
		// Pool is full, close the connection
		conn.Close()
		cp.removeConnectionUnsafe(conn.ID())
		atomic.AddInt64(&cp.metrics.ConnectionsDestroyed, 1)
		return nil
	}
}

// Close shuts down the connection pool
func (cp *ConnectionPool) Close() error {
	close(cp.shutdown)
	cp.workers.Wait()

	cp.mutex.Lock()
	defer cp.mutex.Unlock()

	// Close all connections
	for _, conn := range cp.connections {
		conn.Close()
	}
	
	// Close idle connections channel
	close(cp.idleConnections)
	for conn := range cp.idleConnections {
		conn.Close()
	}

	cp.logger.Info("Connection pool closed")
	return nil
}

// GetMetrics returns current pool metrics
func (cp *ConnectionPool) GetMetrics() PoolMetrics {
	return PoolMetrics{
		TotalConnections:     atomic.LoadInt64(&cp.metrics.TotalConnections),
		ActiveConnections:    atomic.LoadInt64(&cp.metrics.ActiveConnections),
		IdleConnections:      atomic.LoadInt64(&cp.metrics.IdleConnections),
		UnhealthyConnections: atomic.LoadInt64(&cp.metrics.UnhealthyConnections),
		ConnectionsCreated:   atomic.LoadInt64(&cp.metrics.ConnectionsCreated),
		ConnectionsDestroyed: atomic.LoadInt64(&cp.metrics.ConnectionsDestroyed),
		ConnectionErrors:     atomic.LoadInt64(&cp.metrics.ConnectionErrors),
		HealthChecksPassed:   atomic.LoadInt64(&cp.metrics.HealthChecksPassed),
		HealthChecksFailed:   atomic.LoadInt64(&cp.metrics.HealthChecksFailed),
		AverageResponseTime:  cp.metrics.AverageResponseTime,
	}
}

// SetHealthCheckFunc sets a custom health check function
func (cp *ConnectionPool) SetHealthCheckFunc(f func(*Connection) bool) {
	cp.healthCheckFunc = f
}

// createConnection creates a new connection
func (cp *ConnectionPool) createConnection() (*Connection, error) {
	ctx, cancel := context.WithTimeout(context.Background(), cp.config.ConnectionTimeout)
	defer cancel()

	// Apply retry logic
	var conn net.Conn
	var err error

	for attempt := 0; attempt < cp.config.RetryAttempts; attempt++ {
		conn, err = cp.dialFunc()
		if err == nil {
			break
		}

		if attempt < cp.config.RetryAttempts-1 {
			select {
			case <-time.After(cp.config.RetryDelay):
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	if err != nil {
		return nil, fmt.Errorf("failed to establish connection after %d attempts: %w", cp.config.RetryAttempts, err)
	}

	// Configure connection
	if tcpConn, ok := conn.(*net.TCPConn); ok {
		if cp.config.KeepAlive > 0 {
			tcpConn.SetKeepAlive(true)
			tcpConn.SetKeepAlivePeriod(cp.config.KeepAlive)
		}
		if cp.config.NoDelay {
			tcpConn.SetNoDelay(true)
		}
	}

	// Wrap in TLS if enabled
	if cp.config.EnableTLS {
		tlsConn := tls.Client(conn, cp.config.TLSConfig)
		if err := tlsConn.Handshake(); err != nil {
			conn.Close()
			return nil, fmt.Errorf("TLS handshake failed: %w", err)
		}
		conn = tlsConn
	}

	// Create connection wrapper
	id := fmt.Sprintf("conn-%d", time.Now().UnixNano())
	poolConn := NewConnection(conn, id, cp)

	cp.mutex.Lock()
	cp.connections[id] = poolConn
	atomic.AddInt64(&cp.metrics.TotalConnections, 1)
	cp.mutex.Unlock()

	cp.logger.Debug("Created new connection", zap.String("id", id))
	return poolConn, nil
}

// moveToActive moves a connection to the active pool
func (cp *ConnectionPool) moveToActive(conn *Connection) {
	cp.mutex.Lock()
	defer cp.mutex.Unlock()
	cp.activeConns[conn.ID()] = conn
}

// removeConnection removes a connection from all pools
func (cp *ConnectionPool) removeConnection(id string) {
	cp.mutex.Lock()
	defer cp.mutex.Unlock()
	cp.removeConnectionUnsafe(id)
}

// removeConnectionUnsafe removes a connection without locking (must be called with lock held)
func (cp *ConnectionPool) removeConnectionUnsafe(id string) {
	if conn, exists := cp.connections[id]; exists {
		delete(cp.connections, id)
		delete(cp.activeConns, id)
		atomic.AddInt64(&cp.metrics.TotalConnections, -1)
		cp.logger.Debug("Removed connection", zap.String("id", conn.ID()))
	}
}

// warmUp pre-populates the pool with minimum connections
func (cp *ConnectionPool) warmUp() {
	for i := 0; i < cp.config.MinConnections; i++ {
		conn, err := cp.createConnection()
		if err != nil {
			cp.logger.Warn("Failed to create connection during warmup", zap.Error(err))
			continue
		}

		conn.Release()
		select {
		case cp.idleConnections <- conn:
			atomic.AddInt64(&cp.metrics.IdleConnections, 1)
		default:
			conn.Close()
			cp.removeConnection(conn.ID())
		}
	}

	cp.logger.Info("Pool warmed up", zap.Int("connections", cp.config.MinConnections))
}

// startBackgroundWorkers starts background maintenance tasks
func (cp *ConnectionPool) startBackgroundWorkers() {
	// Health checker
	cp.workers.Add(1)
	go cp.healthChecker()

	// Connection reaper
	cp.workers.Add(1)
	go cp.connectionReaper()

	// Metrics collector
	cp.workers.Add(1)
	go cp.metricsCollector()
}

// healthChecker periodically checks connection health
func (cp *ConnectionPool) healthChecker() {
	defer cp.workers.Done()
	ticker := time.NewTicker(cp.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cp.performHealthChecks()
		case <-cp.shutdown:
			return
		}
	}
}

// performHealthChecks checks the health of all connections
func (cp *ConnectionPool) performHealthChecks() {
	cp.mutex.RLock()
	connections := make([]*Connection, 0, len(cp.connections))
	for _, conn := range cp.connections {
		connections = append(connections, conn)
	}
	cp.mutex.RUnlock()

	for _, conn := range connections {
		if conn.State() == StateActive {
			continue // Skip active connections
		}

		healthy := cp.healthCheckFunc(conn)
		if healthy {
			conn.UpdateHealthScore(1.0)
			atomic.AddInt64(&cp.metrics.HealthChecksPassed, 1)
		} else {
			conn.UpdateHealthScore(0.0)
			conn.SetState(StateUnhealthy)
			atomic.AddInt64(&cp.metrics.HealthChecksFailed, 1)
			atomic.AddInt64(&cp.metrics.UnhealthyConnections, 1)
		}
	}
}

// connectionReaper removes old and unhealthy connections
func (cp *ConnectionPool) connectionReaper() {
	defer cp.workers.Done()
	ticker := time.NewTicker(time.Minute * 5)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cp.reapConnections()
		case <-cp.shutdown:
			return
		}
	}
}

// reapConnections removes connections that should be closed
func (cp *ConnectionPool) reapConnections() {
	cp.mutex.Lock()
	defer cp.mutex.Unlock()

	toRemove := make([]string, 0)

	for id, conn := range cp.connections {
		shouldReap := false

		// Check age
		if conn.Age() > cp.config.MaxConnectionAge {
			shouldReap = true
		}

		// Check idle time
		if conn.IdleTime() > cp.config.MaxIdleTime && conn.State() == StateIdle {
			shouldReap = true
		}

		// Check health
		if !conn.IsHealthy() {
			shouldReap = true
		}

		if shouldReap && conn.State() != StateActive {
			toRemove = append(toRemove, id)
			conn.Close()
		}
	}

	for _, id := range toRemove {
		cp.removeConnectionUnsafe(id)
		atomic.AddInt64(&cp.metrics.ConnectionsDestroyed, 1)
	}

	if len(toRemove) > 0 {
		cp.logger.Debug("Reaped connections", zap.Int("count", len(toRemove)))
	}
}

// metricsCollector updates pool metrics
func (cp *ConnectionPool) metricsCollector() {
	defer cp.workers.Done()
	ticker := time.NewTicker(time.Second * 30)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cp.updateMetrics()
		case <-cp.shutdown:
			return
		}
	}
}

// updateMetrics calculates and updates pool metrics
func (cp *ConnectionPool) updateMetrics() {
	cp.mutex.RLock()
	totalConnections := len(cp.connections)
	activeConnections := len(cp.activeConns)
	idleConnections := len(cp.idleConnections)
	cp.mutex.RUnlock()

	atomic.StoreInt64(&cp.metrics.TotalConnections, int64(totalConnections))
	atomic.StoreInt64(&cp.metrics.ActiveConnections, int64(activeConnections))
	atomic.StoreInt64(&cp.metrics.IdleConnections, int64(idleConnections))
}

// defaultHealthCheck performs a basic health check on a connection
func defaultHealthCheck(conn *Connection) bool {
	// Set a short deadline for the health check
	deadline := time.Now().Add(time.Second * 5)
	conn.SetDeadline(deadline)
	defer conn.SetDeadline(time.Time{})

	// Try to write a small amount of data (ping)
	_, err := conn.Write([]byte("PING\n"))
	if err != nil {
		return false
	}

	// Try to read the response
	buffer := make([]byte, 1024)
	_, err = conn.Read(buffer)
	return err == nil
}