package compress

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/s2"
	"github.com/klauspost/compress/zstd"
	"github.com/vmihailenco/msgpack/v5"
	"go.uber.org/zap"
)

// CompressionType represents different compression algorithms
type CompressionType int

const (
	NoCompression CompressionType = iota
	Gzip
	Deflate
	Snappy
	LZ4
	Zstandard
	Brotli
)

// String returns string representation of compression type
func (ct CompressionType) String() string {
	switch ct {
	case NoCompression:
		return "none"
	case Gzip:
		return "gzip"
	case Deflate:
		return "deflate"
	case Snappy:
		return "snappy"
	case LZ4:
		return "lz4"
	case Zstandard:
		return "zstd"
	case Brotli:
		return "brotli"
	default:
		return "unknown"
	}
}

// CompressionLevel represents compression quality levels
type CompressionLevel int

const (
	LevelFastest CompressionLevel = iota
	LevelDefault
	LevelBetter
	LevelBest
)

// SerializationFormat represents different serialization formats
type SerializationFormat int

const (
	JSON SerializationFormat = iota
	MessagePack
	Protobuf
	Binary
)

// String returns string representation of serialization format
func (sf SerializationFormat) String() string {
	switch sf {
	case JSON:
		return "json"
	case MessagePack:
		return "msgpack"
	case Protobuf:
		return "protobuf"
	case Binary:
		return "binary"
	default:
		return "unknown"
	}
}

// CompressionMetrics tracks compression performance
type CompressionMetrics struct {
	TotalOperations   int64   `json:"total_operations"`
	CompressionOps    int64   `json:"compression_ops"`
	DecompressionOps  int64   `json:"decompression_ops"`
	TotalBytesIn      int64   `json:"total_bytes_in"`
	TotalBytesOut     int64   `json:"total_bytes_out"`
	CompressionRatio  float64 `json:"compression_ratio"`
	AverageTime       time.Duration `json:"average_time"`
	ThroughputMBps    float64 `json:"throughput_mbps"`
	ErrorCount        int64   `json:"error_count"`
}

// CompressorConfig holds configuration for the compressor
type CompressorConfig struct {
	DefaultType          CompressionType     `json:"default_type"`
	DefaultLevel         CompressionLevel    `json:"default_level"`
	SerializationFormat  SerializationFormat `json:"serialization_format"`
	MinSizeThreshold     int                 `json:"min_size_threshold"`     // Don't compress below this size
	MaxSizeThreshold     int                 `json:"max_size_threshold"`     // Don't compress above this size
	EnablePooling        bool                `json:"enable_pooling"`         // Use buffer pools
	PoolSize             int                 `json:"pool_size"`              // Size of buffer pools
	EnableMetrics        bool                `json:"enable_metrics"`         // Track metrics
	EnableAsyncMode      bool                `json:"enable_async_mode"`      // Enable async compression
	WorkerCount          int                 `json:"worker_count"`           // Number of compression workers
	QueueSize            int                 `json:"queue_size"`             // Async queue size
	AdaptiveCompression  bool                `json:"adaptive_compression"`   // Choose algorithm based on data
	EnableDeltaEncoding  bool                `json:"enable_delta_encoding"`  // Enable delta compression
	DictionarySize       int                 `json:"dictionary_size"`        // Dictionary size for adaptive compression
}

// DefaultCompressorConfig returns a default compressor configuration
func DefaultCompressorConfig() *CompressorConfig {
	return &CompressorConfig{
		DefaultType:         Zstandard,
		DefaultLevel:        LevelDefault,
		SerializationFormat: MessagePack,
		MinSizeThreshold:    64,    // 64 bytes
		MaxSizeThreshold:    1 << 20, // 1MB
		EnablePooling:       true,
		PoolSize:           50,
		EnableMetrics:      true,
		EnableAsyncMode:    false,
		WorkerCount:        4,
		QueueSize:          1000,
		AdaptiveCompression: true,
		EnableDeltaEncoding: false,
		DictionarySize:     8192, // 8KB
	}
}

// CompressedData represents compressed data with metadata
type CompressedData struct {
	Data               []byte              `json:"data"`
	OriginalSize       int                 `json:"original_size"`
	CompressedSize     int                 `json:"compressed_size"`
	CompressionType    CompressionType     `json:"compression_type"`
	CompressionLevel   CompressionLevel    `json:"compression_level"`
	SerializationFormat SerializationFormat `json:"serialization_format"`
	CompressionRatio   float64             `json:"compression_ratio"`
	Timestamp          time.Time           `json:"timestamp"`
	Checksum           uint32              `json:"checksum"`
	IsDelta            bool                `json:"is_delta"`
}

// CompressionTask represents an async compression task
type CompressionTask struct {
	ID       string
	Data     interface{}
	Type     CompressionType
	Level    CompressionLevel
	Format   SerializationFormat
	ResultCh chan CompressionResult
	Ctx      interface{}
}

// CompressionResult represents the result of a compression operation
type CompressionResult struct {
	ID           string
	Compressed   *CompressedData
	Error        error
	Duration     time.Duration
	Ctx          interface{}
}

// BufferPool manages reusable byte buffers
type BufferPool struct {
	pool sync.Pool
}

// NewBufferPool creates a new buffer pool
func NewBufferPool(initialSize int) *BufferPool {
	return &BufferPool{
		pool: sync.Pool{
			New: func() interface{} {
				return bytes.NewBuffer(make([]byte, 0, initialSize))
			},
		},
	}
}

// Get retrieves a buffer from the pool
func (bp *BufferPool) Get() *bytes.Buffer {
	buf := bp.pool.Get().(*bytes.Buffer)
	buf.Reset()
	return buf
}

// Put returns a buffer to the pool
func (bp *BufferPool) Put(buf *bytes.Buffer) {
	if buf.Cap() > 1<<20 { // Don't pool buffers larger than 1MB
		return
	}
	bp.pool.Put(buf)
}

// DeltaEncoder implements delta encoding for similar data
type DeltaEncoder struct {
	previousData map[string][]byte
	mutex        sync.RWMutex
}

// NewDeltaEncoder creates a new delta encoder
func NewDeltaEncoder() *DeltaEncoder {
	return &DeltaEncoder{
		previousData: make(map[string][]byte),
	}
}

// Encode creates a delta from the previous version
func (de *DeltaEncoder) Encode(key string, data []byte) ([]byte, bool) {
	de.mutex.Lock()
	defer de.mutex.Unlock()

	previous, exists := de.previousData[key]
	if !exists {
		de.previousData[key] = make([]byte, len(data))
		copy(de.previousData[key], data)
		return data, false // First time, return original data
	}

	// Simple delta encoding - calculate differences
	delta := make([]byte, 0, len(data))
	for i := 0; i < len(data) && i < len(previous); i++ {
		delta = append(delta, data[i]^previous[i])
	}

	// Append new data if current is longer
	if len(data) > len(previous) {
		delta = append(delta, data[len(previous):]...)
	}

	// Update stored data
	de.previousData[key] = make([]byte, len(data))
	copy(de.previousData[key], data)

	return delta, true
}

// Decode reconstructs data from delta
func (de *DeltaEncoder) Decode(key string, delta []byte) ([]byte, error) {
	de.mutex.RLock()
	previous, exists := de.previousData[key]
	de.mutex.RUnlock()

	if !exists {
		return delta, nil // First time, delta is the original data
	}

	// Reconstruct data
	data := make([]byte, len(delta))
	for i := 0; i < len(delta) && i < len(previous); i++ {
		data[i] = delta[i] ^ previous[i]
	}

	// Append additional data if delta is longer
	if len(delta) > len(previous) {
		data = append(data[:len(previous)], delta[len(previous):]...)
	}

	return data, nil
}

// RealTimeCompressor implements high-performance real-time compression
type RealTimeCompressor struct {
	config        *CompressorConfig
	metrics       *CompressionMetrics
	bufferPool    *BufferPool
	deltaEncoder  *DeltaEncoder
	zstdEncoder   *zstd.Encoder
	zstdDecoder   *zstd.Decoder
	logger        *zap.Logger
	
	// Async processing
	taskQueue     chan CompressionTask
	workers       []chan CompressionTask
	shutdown      chan struct{}
	workerGroup   sync.WaitGroup
	
	// Adaptive compression
	algorithmStats map[CompressionType]*AlgorithmStats
	statsMutex     sync.RWMutex
}

// AlgorithmStats tracks performance stats for each compression algorithm
type AlgorithmStats struct {
	TotalOps         int64
	TotalTimeNs      int64
	TotalBytesIn     int64
	TotalBytesOut    int64
	AverageRatio     float64
	AverageThroughput float64
}

// NewRealTimeCompressor creates a new real-time compressor
func NewRealTimeCompressor(config *CompressorConfig, logger *zap.Logger) (*RealTimeCompressor, error) {
	if config == nil {
		config = DefaultCompressorConfig()
	}

	// Initialize zstd encoder/decoder
	zstdEncoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd encoder: %w", err)
	}

	zstdDecoder, err := zstd.NewReader(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd decoder: %w", err)
	}

	compressor := &RealTimeCompressor{
		config:         config,
		metrics:        &CompressionMetrics{},
		bufferPool:     NewBufferPool(4096),
		deltaEncoder:   NewDeltaEncoder(),
		zstdEncoder:    zstdEncoder,
		zstdDecoder:    zstdDecoder,
		logger:         logger,
		shutdown:       make(chan struct{}),
		algorithmStats: make(map[CompressionType]*AlgorithmStats),
	}

	// Initialize algorithm stats
	compressor.initAlgorithmStats()

	// Start async workers if enabled
	if config.EnableAsyncMode {
		compressor.startAsyncWorkers()
	}

	return compressor, nil
}

// Compress compresses data using the specified or default algorithm
func (rtc *RealTimeCompressor) Compress(data interface{}, compressionType CompressionType, level CompressionLevel) (*CompressedData, error) {
	start := time.Now()
	
	// Serialize data first
	serialized, err := rtc.serialize(data, rtc.config.SerializationFormat)
	if err != nil {
		atomic.AddInt64(&rtc.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("serialization failed: %w", err)
	}

	originalSize := len(serialized)
	
	// Check size thresholds
	if originalSize < rtc.config.MinSizeThreshold || originalSize > rtc.config.MaxSizeThreshold {
		// Return uncompressed data
		return &CompressedData{
			Data:               serialized,
			OriginalSize:       originalSize,
			CompressedSize:     originalSize,
			CompressionType:    NoCompression,
			CompressionLevel:   level,
			SerializationFormat: rtc.config.SerializationFormat,
			CompressionRatio:   1.0,
			Timestamp:          time.Now(),
			Checksum:           rtc.calculateChecksum(serialized),
		}, nil
	}

	// Choose compression algorithm adaptively if enabled
	if rtc.config.AdaptiveCompression && compressionType == rtc.config.DefaultType {
		compressionType = rtc.chooseOptimalAlgorithm(serialized)
	}

	// Perform compression
	compressed, err := rtc.compressData(serialized, compressionType, level)
	if err != nil {
		atomic.AddInt64(&rtc.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("compression failed: %w", err)
	}

	compressedSize := len(compressed)
	compressionRatio := float64(originalSize) / float64(compressedSize)

	// Update metrics
	duration := time.Since(start)
	rtc.updateMetrics(true, originalSize, compressedSize, duration)
	rtc.updateAlgorithmStats(compressionType, duration, originalSize, compressedSize)

	result := &CompressedData{
		Data:               compressed,
		OriginalSize:       originalSize,
		CompressedSize:     compressedSize,
		CompressionType:    compressionType,
		CompressionLevel:   level,
		SerializationFormat: rtc.config.SerializationFormat,
		CompressionRatio:   compressionRatio,
		Timestamp:          time.Now(),
		Checksum:           rtc.calculateChecksum(serialized),
	}

	rtc.logger.Debug("Compression completed",
		zap.String("algorithm", compressionType.String()),
		zap.Int("original_size", originalSize),
		zap.Int("compressed_size", compressedSize),
		zap.Float64("ratio", compressionRatio),
		zap.Duration("duration", duration))

	return result, nil
}

// Decompress decompresses previously compressed data
func (rtc *RealTimeCompressor) Decompress(compressedData *CompressedData) (interface{}, error) {
	start := time.Now()

	// Handle uncompressed data
	if compressedData.CompressionType == NoCompression {
		return rtc.deserialize(compressedData.Data, compressedData.SerializationFormat)
	}

	// Decompress data
	decompressed, err := rtc.decompressData(compressedData.Data, compressedData.CompressionType)
	if err != nil {
		atomic.AddInt64(&rtc.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("decompression failed: %w", err)
	}

	// Verify checksum
	if rtc.calculateChecksum(decompressed) != compressedData.Checksum {
		atomic.AddInt64(&rtc.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("checksum verification failed")
	}

	// Deserialize
	result, err := rtc.deserialize(decompressed, compressedData.SerializationFormat)
	if err != nil {
		atomic.AddInt64(&rtc.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("deserialization failed: %w", err)
	}

	// Update metrics
	duration := time.Since(start)
	rtc.updateMetrics(false, len(decompressed), len(compressedData.Data), duration)

	rtc.logger.Debug("Decompression completed",
		zap.String("algorithm", compressedData.CompressionType.String()),
		zap.Duration("duration", duration))

	return result, nil
}

// CompressAsync performs asynchronous compression
func (rtc *RealTimeCompressor) CompressAsync(id string, data interface{}, compressionType CompressionType, level CompressionLevel, ctx interface{}) chan CompressionResult {
	resultCh := make(chan CompressionResult, 1)

	if !rtc.config.EnableAsyncMode {
		// Fallback to synchronous compression
		go func() {
			defer close(resultCh)
			compressed, err := rtc.Compress(data, compressionType, level)
			resultCh <- CompressionResult{
				ID:         id,
				Compressed: compressed,
				Error:      err,
				Ctx:        ctx,
			}
		}()
		return resultCh
	}

	task := CompressionTask{
		ID:       id,
		Data:     data,
		Type:     compressionType,
		Level:    level,
		Format:   rtc.config.SerializationFormat,
		ResultCh: resultCh,
		Ctx:      ctx,
	}

	select {
	case rtc.taskQueue <- task:
		// Task queued successfully
	default:
		// Queue is full, handle synchronously
		go func() {
			defer close(resultCh)
			compressed, err := rtc.Compress(data, compressionType, level)
			resultCh <- CompressionResult{
				ID:         id,
				Compressed: compressed,
				Error:      err,
				Ctx:        ctx,
			}
		}()
	}

	return resultCh
}

// GetMetrics returns current compression metrics
func (rtc *RealTimeCompressor) GetMetrics() CompressionMetrics {
	totalOps := atomic.LoadInt64(&rtc.metrics.TotalOperations)
	totalBytesIn := atomic.LoadInt64(&rtc.metrics.TotalBytesIn)
	totalBytesOut := atomic.LoadInt64(&rtc.metrics.TotalBytesOut)

	ratio := 1.0
	if totalBytesOut > 0 {
		ratio = float64(totalBytesIn) / float64(totalBytesOut)
	}

	return CompressionMetrics{
		TotalOperations:  totalOps,
		CompressionOps:   atomic.LoadInt64(&rtc.metrics.CompressionOps),
		DecompressionOps: atomic.LoadInt64(&rtc.metrics.DecompressionOps),
		TotalBytesIn:     totalBytesIn,
		TotalBytesOut:    totalBytesOut,
		CompressionRatio: ratio,
		AverageTime:      rtc.metrics.AverageTime,
		ThroughputMBps:   rtc.metrics.ThroughputMBps,
		ErrorCount:       atomic.LoadInt64(&rtc.metrics.ErrorCount),
	}
}

// Close shuts down the compressor
func (rtc *RealTimeCompressor) Close() error {
	close(rtc.shutdown)
	rtc.workerGroup.Wait()

	if rtc.zstdEncoder != nil {
		rtc.zstdEncoder.Close()
	}
	if rtc.zstdDecoder != nil {
		rtc.zstdDecoder.Close()
	}

	rtc.logger.Info("Real-time compressor stopped")
	return nil
}

// Internal methods

func (rtc *RealTimeCompressor) serialize(data interface{}, format SerializationFormat) ([]byte, error) {
	switch format {
	case JSON:
		return json.Marshal(data)
	case MessagePack:
		return msgpack.Marshal(data)
	case Binary:
		// For binary serialization, assume data is already []byte
		if bytes, ok := data.([]byte); ok {
			return bytes, nil
		}
		return nil, fmt.Errorf("binary format requires []byte input")
	default:
		return nil, fmt.Errorf("unsupported serialization format: %s", format)
	}
}

func (rtc *RealTimeCompressor) deserialize(data []byte, format SerializationFormat) (interface{}, error) {
	switch format {
	case JSON:
		var result interface{}
		err := json.Unmarshal(data, &result)
		return result, err
	case MessagePack:
		var result interface{}
		err := msgpack.Unmarshal(data, &result)
		return result, err
	case Binary:
		return data, nil
	default:
		return nil, fmt.Errorf("unsupported serialization format: %s", format)
	}
}

func (rtc *RealTimeCompressor) compressData(data []byte, compressionType CompressionType, level CompressionLevel) ([]byte, error) {
	var buf *bytes.Buffer
	if rtc.config.EnablePooling {
		buf = rtc.bufferPool.Get()
		defer rtc.bufferPool.Put(buf)
	} else {
		buf = &bytes.Buffer{}
	}

	switch compressionType {
	case Gzip:
		return rtc.compressGzip(data, level)
	case Deflate:
		return rtc.compressDeflate(data, level)
	case Snappy:
		return rtc.compressSnappy(data)
	case Zstandard:
		return rtc.compressZstd(data, level)
	default:
		return nil, fmt.Errorf("unsupported compression type: %s", compressionType)
	}
}

func (rtc *RealTimeCompressor) decompressData(data []byte, compressionType CompressionType) ([]byte, error) {
	switch compressionType {
	case Gzip:
		return rtc.decompressGzip(data)
	case Deflate:
		return rtc.decompressDeflate(data)
	case Snappy:
		return rtc.decompressSnappy(data)
	case Zstandard:
		return rtc.decompressZstd(data)
	default:
		return nil, fmt.Errorf("unsupported compression type: %s", compressionType)
	}
}

func (rtc *RealTimeCompressor) compressGzip(data []byte, level CompressionLevel) ([]byte, error) {
	var buf bytes.Buffer
	
	gzipLevel := gzip.DefaultCompression
	switch level {
	case LevelFastest:
		gzipLevel = gzip.BestSpeed
	case LevelBetter:
		gzipLevel = gzip.BestCompression
	case LevelBest:
		gzipLevel = gzip.BestCompression
	}

	writer, err := gzip.NewWriterLevel(&buf, gzipLevel)
	if err != nil {
		return nil, err
	}

	_, err = writer.Write(data)
	if err != nil {
		return nil, err
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (rtc *RealTimeCompressor) decompressGzip(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	return io.ReadAll(reader)
}

func (rtc *RealTimeCompressor) compressDeflate(data []byte, level CompressionLevel) ([]byte, error) {
	var buf bytes.Buffer
	
	deflateLevel := flate.DefaultCompression
	switch level {
	case LevelFastest:
		deflateLevel = flate.BestSpeed
	case LevelBetter:
		deflateLevel = flate.BestCompression
	case LevelBest:
		deflateLevel = flate.BestCompression
	}

	writer, err := flate.NewWriter(&buf, deflateLevel)
	if err != nil {
		return nil, err
	}

	_, err = writer.Write(data)
	if err != nil {
		return nil, err
	}

	err = writer.Close()
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (rtc *RealTimeCompressor) decompressDeflate(data []byte) ([]byte, error) {
	reader := flate.NewReader(bytes.NewReader(data))
	defer reader.Close()

	return io.ReadAll(reader)
}

func (rtc *RealTimeCompressor) compressSnappy(data []byte) ([]byte, error) {
	return s2.Encode(nil, data), nil
}

func (rtc *RealTimeCompressor) decompressSnappy(data []byte) ([]byte, error) {
	return s2.Decode(nil, data)
}

func (rtc *RealTimeCompressor) compressZstd(data []byte, level CompressionLevel) ([]byte, error) {
	return rtc.zstdEncoder.EncodeAll(data, nil), nil
}

func (rtc *RealTimeCompressor) decompressZstd(data []byte) ([]byte, error) {
	return rtc.zstdDecoder.DecodeAll(data, nil)
}

func (rtc *RealTimeCompressor) calculateChecksum(data []byte) uint32 {
	// Simple checksum implementation - in production, use CRC32 or similar
	checksum := uint32(0)
	for _, b := range data {
		checksum = checksum*31 + uint32(b)
	}
	return checksum
}

func (rtc *RealTimeCompressor) chooseOptimalAlgorithm(data []byte) CompressionType {
	rtc.statsMutex.RLock()
	defer rtc.statsMutex.RUnlock()

	// Choose algorithm based on data characteristics and historical performance
	dataSize := len(data)
	
	// For small data, use fast algorithms
	if dataSize < 1024 {
		return Snappy
	}
	
	// For medium data, balance speed and ratio
	if dataSize < 64*1024 {
		// Choose based on best throughput
		bestAlgorithm := Zstandard
		bestThroughput := 0.0
		
		for alg, stats := range rtc.algorithmStats {
			if stats.AverageThroughput > bestThroughput {
				bestThroughput = stats.AverageThroughput
				bestAlgorithm = alg
			}
		}
		return bestAlgorithm
	}
	
	// For large data, prioritize compression ratio
	bestAlgorithm := Zstandard
	bestRatio := 1.0
	
	for alg, stats := range rtc.algorithmStats {
		if stats.AverageRatio > bestRatio {
			bestRatio = stats.AverageRatio
			bestAlgorithm = alg
		}
	}
	
	return bestAlgorithm
}

func (rtc *RealTimeCompressor) updateMetrics(isCompression bool, bytesIn, bytesOut int, duration time.Duration) {
	atomic.AddInt64(&rtc.metrics.TotalOperations, 1)
	atomic.AddInt64(&rtc.metrics.TotalBytesIn, int64(bytesIn))
	atomic.AddInt64(&rtc.metrics.TotalBytesOut, int64(bytesOut))

	if isCompression {
		atomic.AddInt64(&rtc.metrics.CompressionOps, 1)
	} else {
		atomic.AddInt64(&rtc.metrics.DecompressionOps, 1)
	}

	// Update throughput (simplified)
	if duration > 0 {
		mbps := float64(bytesIn) / (1024 * 1024) / duration.Seconds()
		rtc.metrics.ThroughputMBps = mbps
	}
}

func (rtc *RealTimeCompressor) updateAlgorithmStats(algorithm CompressionType, duration time.Duration, bytesIn, bytesOut int) {
	rtc.statsMutex.Lock()
	defer rtc.statsMutex.Unlock()

	stats, exists := rtc.algorithmStats[algorithm]
	if !exists {
		stats = &AlgorithmStats{}
		rtc.algorithmStats[algorithm] = stats
	}

	stats.TotalOps++
	stats.TotalTimeNs += duration.Nanoseconds()
	stats.TotalBytesIn += int64(bytesIn)
	stats.TotalBytesOut += int64(bytesOut)
	
	// Update averages
	if stats.TotalBytesOut > 0 {
		stats.AverageRatio = float64(stats.TotalBytesIn) / float64(stats.TotalBytesOut)
	}
	
	if stats.TotalTimeNs > 0 {
		avgTimeSeconds := float64(stats.TotalTimeNs) / float64(time.Second) / float64(stats.TotalOps)
		stats.AverageThroughput = (float64(stats.TotalBytesIn) / (1024 * 1024)) / (avgTimeSeconds * float64(stats.TotalOps))
	}
}

func (rtc *RealTimeCompressor) initAlgorithmStats() {
	algorithms := []CompressionType{Gzip, Deflate, Snappy, Zstandard}
	for _, alg := range algorithms {
		rtc.algorithmStats[alg] = &AlgorithmStats{}
	}
}

func (rtc *RealTimeCompressor) startAsyncWorkers() {
	rtc.taskQueue = make(chan CompressionTask, rtc.config.QueueSize)
	rtc.workers = make([]chan CompressionTask, rtc.config.WorkerCount)

	for i := 0; i < rtc.config.WorkerCount; i++ {
		workerCh := make(chan CompressionTask, 10)
		rtc.workers[i] = workerCh
		
		rtc.workerGroup.Add(1)
		go rtc.compressionWorker(i, workerCh)
	}

	// Task distributor
	rtc.workerGroup.Add(1)
	go rtc.taskDistributor()
}

func (rtc *RealTimeCompressor) taskDistributor() {
	defer rtc.workerGroup.Done()
	
	workerIndex := 0
	for {
		select {
		case task := <-rtc.taskQueue:
			// Distribute to next worker
			worker := rtc.workers[workerIndex]
			workerIndex = (workerIndex + 1) % len(rtc.workers)
			
			select {
			case worker <- task:
				// Task sent to worker
			default:
				// Worker is busy, handle synchronously
				go rtc.handleCompressionTask(task)
			}
		case <-rtc.shutdown:
			// Close all worker channels
			for _, worker := range rtc.workers {
				close(worker)
			}
			return
		}
	}
}

func (rtc *RealTimeCompressor) compressionWorker(id int, taskCh chan CompressionTask) {
	defer rtc.workerGroup.Done()
	
	rtc.logger.Debug("Compression worker started", zap.Int("worker_id", id))
	
	for task := range taskCh {
		rtc.handleCompressionTask(task)
	}
	
	rtc.logger.Debug("Compression worker stopped", zap.Int("worker_id", id))
}

func (rtc *RealTimeCompressor) handleCompressionTask(task CompressionTask) {
	start := time.Now()
	
	compressed, err := rtc.Compress(task.Data, task.Type, task.Level)
	duration := time.Since(start)
	
	result := CompressionResult{
		ID:         task.ID,
		Compressed: compressed,
		Error:      err,
		Duration:   duration,
		Ctx:        task.Ctx,
	}
	
	task.ResultCh <- result
	close(task.ResultCh)
}