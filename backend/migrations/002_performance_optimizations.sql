-- Migration 002: Performance Optimizations
-- Created: 2024-01-02
-- Description: Additional indexes and optimizations for heavy workloads

-- Additional composite indexes for common query patterns
CREATE INDEX idx_chunks_doc_page ON chunks(document_id, page_number);
CREATE INDEX idx_chunks_embedding_exists ON chunks(document_id) WHERE embedding_vector IS NOT NULL;
CREATE INDEX idx_detections_doc_confidence ON detections(document_id, confidence_score DESC);
CREATE INDEX idx_detections_type_created ON detections(detection_type, created_at DESC);
CREATE INDEX idx_patterns_category_effectiveness ON patterns(category, effectiveness_score DESC);

-- Partial indexes for active/processed records
CREATE INDEX idx_documents_processed ON documents(id, last_processed) WHERE processing_status = 'completed';
CREATE INDEX idx_patterns_active_effective ON patterns(id, effectiveness_score) WHERE is_active = TRUE AND effectiveness_score >= 0.7;
CREATE INDEX idx_detections_high_confidence ON detections(id, confidence_score, detection_type) WHERE confidence_score >= 0.8;

-- Covering indexes for read-heavy queries
CREATE INDEX idx_documents_list_view ON documents(processing_status, upload_date DESC, id, filename, file_type, total_chunks);
CREATE INDEX idx_detections_summary ON detections(document_id, detection_type, confidence_score, matched_text, page_number);

-- Text search optimization (if using FTS)
-- CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, section_title, content='chunks', content_rowid='id');

-- Insert triggers for FTS (uncomment if using FTS)
-- CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
--     INSERT INTO chunks_fts(rowid, content, section_title) VALUES (new.id, new.content, new.section_title);
-- END;

-- CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
--     DELETE FROM chunks_fts WHERE rowid = old.id;
-- END;

-- CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
--     DELETE FROM chunks_fts WHERE rowid = old.id;
--     INSERT INTO chunks_fts(rowid, content, section_title) VALUES (new.id, new.content, new.section_title);
-- END;

-- Materialized view for analytics (SQLite doesn't support materialized views, so we'll use a table)
CREATE TABLE analytics_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    dimensions JSON,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_analytics_cache_metric ON analytics_cache(metric_name, expires_at);
CREATE INDEX idx_analytics_cache_expires ON analytics_cache(expires_at);

-- Function to calculate detection statistics (stored procedure equivalent)
-- We'll implement this in the application layer since SQLite doesn't support stored procedures

-- Performance monitoring table
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type VARCHAR(100) NOT NULL,
    duration_ms INTEGER NOT NULL,
    record_count INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_metrics_operation ON performance_metrics(operation_type, created_at);
CREATE INDEX idx_performance_metrics_duration ON performance_metrics(duration_ms);

-- Trigger to clean up old performance metrics (keep only last 30 days)
CREATE TRIGGER cleanup_old_metrics
    AFTER INSERT ON performance_metrics
    BEGIN
        DELETE FROM performance_metrics 
        WHERE created_at < datetime('now', '-30 days');
    END;

-- Configuration table for system settings
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    data_type VARCHAR(50) NOT NULL DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration
INSERT INTO system_config (key, value, data_type, description) VALUES
('embedding_dimension', '1536', 'integer', 'Default embedding vector dimension'),
('similarity_threshold', '0.7', 'float', 'Default similarity threshold for searches'),
('batch_size', '1000', 'integer', 'Default batch size for bulk operations'),
('cache_ttl_documents', '7200', 'integer', 'Cache TTL for documents in seconds'),
('cache_ttl_searches', '900', 'integer', 'Cache TTL for search results in seconds'),
('max_chunks_per_document', '10000', 'integer', 'Maximum chunks allowed per document'),
('min_chunk_length', '100', 'integer', 'Minimum chunk length in characters'),
('max_chunk_length', '8000', 'integer', 'Maximum chunk length in characters'),
('detection_confidence_threshold', '0.5', 'float', 'Minimum confidence for storing detections'),
('enable_auto_validation', 'false', 'boolean', 'Enable automatic validation of high-confidence detections');

-- Optimization: Pre-compute common aggregations
CREATE TABLE document_summaries (
    document_id INTEGER PRIMARY KEY,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    total_detections INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL,
    max_confidence REAL,
    high_confidence_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Trigger to maintain document summaries
CREATE TRIGGER update_document_summary_on_detection
    AFTER INSERT ON detections
    BEGIN
        INSERT OR REPLACE INTO document_summaries (
            document_id, 
            total_chunks, 
            total_detections, 
            avg_confidence, 
            max_confidence,
            high_confidence_count,
            last_updated
        )
        SELECT 
            NEW.document_id,
            COUNT(DISTINCT c.id),
            COUNT(d.id),
            AVG(d.confidence_score),
            MAX(d.confidence_score),
            SUM(CASE WHEN d.confidence_score >= 0.8 THEN 1 ELSE 0 END),
            CURRENT_TIMESTAMP
        FROM chunks c
        LEFT JOIN detections d ON c.id = d.chunk_id
        WHERE c.document_id = NEW.document_id;
    END;

-- Batch processing queue table
CREATE TABLE processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type VARCHAR(100) NOT NULL,
    task_data JSON NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CHECK (priority BETWEEN 1 AND 10),
    CHECK (retry_count >= 0),
    CHECK (max_retries >= 0)
);

CREATE INDEX idx_processing_queue_status ON processing_queue(status, priority DESC, created_at);
CREATE INDEX idx_processing_queue_type ON processing_queue(task_type, status);

-- Health check table for monitoring
CREATE TABLE health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    metadata JSON,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (status IN ('healthy', 'degraded', 'unhealthy'))
);

CREATE INDEX idx_health_checks_name ON health_checks(check_name, checked_at DESC);

-- Database statistics table for monitoring
CREATE TABLE db_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name VARCHAR(100) NOT NULL,
    row_count INTEGER NOT NULL,
    size_bytes INTEGER,
    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to update statistics after significant changes
CREATE TRIGGER update_db_stats_on_bulk_insert
    AFTER INSERT ON chunks
    WHEN (SELECT COUNT(*) FROM chunks WHERE document_id = NEW.document_id) % 1000 = 0
    BEGIN
        INSERT OR REPLACE INTO db_statistics (table_name, row_count, last_analyzed)
        VALUES ('chunks', (SELECT COUNT(*) FROM chunks), CURRENT_TIMESTAMP);
    END;

-- Record migration
INSERT INTO schema_migrations (version, description, checksum) VALUES 
(2, 'Performance optimizations with composite indexes and monitoring', 'b2c3d4e5f6g7');

-- Update table statistics
ANALYZE documents;
ANALYZE chunks;
ANALYZE detections;
ANALYZE patterns;