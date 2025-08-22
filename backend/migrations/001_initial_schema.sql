-- Migration 001: Initial Schema for Arbitration Detection RAG System
-- Created: 2024-01-01
-- Description: Creates initial tables with optimized indexes for RAG system

-- Enable foreign key constraints and performance optimizations
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = memory;

-- Migration metadata table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    checksum VARCHAR(64)
);

-- Documents table: Store original documents and metadata
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_processed TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    total_pages INTEGER,
    total_chunks INTEGER DEFAULT 0,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (file_size > 0),
    CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- Chunks table: Store document chunks with embeddings
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    chunk_hash VARCHAR(64) NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    embedding_vector BLOB,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'text-embedding-ada-002',
    similarity_threshold REAL DEFAULT 0.7,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (content_length > 0),
    CHECK (chunk_index >= 0),
    CHECK (similarity_threshold >= 0 AND similarity_threshold <= 1),
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Patterns table: Store arbitration clause patterns
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_name VARCHAR(200) NOT NULL,
    pattern_text TEXT NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    effectiveness_score REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (effectiveness_score >= 0 AND effectiveness_score <= 1),
    CHECK (pattern_type IN ('regex', 'keyword', 'semantic')),
    CHECK (usage_count >= 0)
);

-- Detections table: Store detection results
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL,
    document_id INTEGER NOT NULL,
    detection_type VARCHAR(100) NOT NULL,
    confidence_score REAL NOT NULL,
    pattern_id INTEGER,
    matched_text TEXT NOT NULL,
    context_before TEXT,
    context_after TEXT,
    start_position INTEGER,
    end_position INTEGER,
    page_number INTEGER,
    detection_method VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    is_validated BOOLEAN DEFAULT FALSE,
    validation_score REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CHECK (detection_method IN ('rule_based', 'ml_model', 'hybrid')),
    CHECK (start_position >= 0),
    CHECK (end_position >= start_position),
    
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
);

-- Query cache table: Store frequently accessed results
CREATE TABLE query_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    query_hash VARCHAR(64) NOT NULL,
    result_data JSON NOT NULL,
    result_count INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 1,
    ttl INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    
    -- Constraints
    CHECK (ttl > 0),
    CHECK (access_count > 0)
);

-- Create optimized indexes
-- Documents indexes
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_hash ON documents(content_hash);
CREATE INDEX idx_documents_type ON documents(file_type);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);
CREATE INDEX idx_documents_filename ON documents(filename);

-- Chunks indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_hash ON chunks(chunk_hash);
CREATE INDEX idx_chunks_length ON chunks(content_length);
CREATE INDEX idx_chunks_page ON chunks(page_number);
CREATE INDEX idx_chunks_doc_index ON chunks(document_id, chunk_index);

-- Detections indexes
CREATE INDEX idx_detections_chunk_id ON detections(chunk_id);
CREATE INDEX idx_detections_document_id ON detections(document_id);
CREATE INDEX idx_detections_type ON detections(detection_type);
CREATE INDEX idx_detections_confidence ON detections(confidence_score);
CREATE INDEX idx_detections_validated ON detections(is_validated);
CREATE INDEX idx_detections_created ON detections(created_at);
CREATE INDEX idx_detections_doc_type ON detections(document_id, detection_type);
CREATE INDEX idx_detections_type_confidence ON detections(detection_type, confidence_score);
CREATE INDEX idx_detections_pattern_confidence ON detections(pattern_id, confidence_score);

-- Patterns indexes
CREATE INDEX idx_patterns_type ON patterns(pattern_type);
CREATE INDEX idx_patterns_category ON patterns(category);
CREATE INDEX idx_patterns_active ON patterns(is_active);
CREATE INDEX idx_patterns_effectiveness ON patterns(effectiveness_score);
CREATE INDEX idx_patterns_usage ON patterns(usage_count);

-- Cache indexes
CREATE INDEX idx_cache_key ON query_cache(cache_key);
CREATE INDEX idx_cache_expires ON query_cache(expires_at);
CREATE INDEX idx_cache_access_count ON query_cache(access_count);

-- Create triggers for automatic updates
CREATE TRIGGER documents_updated_at 
    AFTER UPDATE ON documents
    BEGIN
        UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER patterns_updated_at 
    AFTER UPDATE ON patterns
    BEGIN
        UPDATE patterns SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER pattern_usage_update
    AFTER INSERT ON detections
    WHEN NEW.pattern_id IS NOT NULL
    BEGIN
        UPDATE patterns 
        SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP 
        WHERE id = NEW.pattern_id;
    END;

CREATE TRIGGER update_document_chunk_count
    AFTER INSERT ON chunks
    BEGIN
        UPDATE documents 
        SET total_chunks = (
            SELECT COUNT(*) FROM chunks WHERE document_id = NEW.document_id
        ) 
        WHERE id = NEW.document_id;
    END;

CREATE TRIGGER cache_cleanup
    AFTER INSERT ON query_cache
    BEGIN
        DELETE FROM query_cache WHERE expires_at < CURRENT_TIMESTAMP;
    END;

-- Create views for common queries
CREATE VIEW v_document_stats AS
SELECT 
    d.id,
    d.filename,
    d.file_type,
    d.processing_status,
    d.total_chunks,
    COUNT(DISTINCT det.id) as detection_count,
    AVG(det.confidence_score) as avg_confidence,
    MAX(det.confidence_score) as max_confidence,
    d.upload_date,
    d.last_processed
FROM documents d
LEFT JOIN detections det ON d.id = det.document_id
GROUP BY d.id;

CREATE VIEW v_high_confidence_detections AS
SELECT 
    det.*,
    d.filename,
    d.file_type,
    c.content,
    p.pattern_name,
    p.category
FROM detections det
JOIN documents d ON det.document_id = d.id
JOIN chunks c ON det.chunk_id = c.id
LEFT JOIN patterns p ON det.pattern_id = p.id
WHERE det.confidence_score >= 0.8
ORDER BY det.confidence_score DESC;

-- Insert default patterns
INSERT INTO patterns (pattern_name, pattern_text, pattern_type, category, effectiveness_score) VALUES
('Mandatory Arbitration Clause', 'any claim.*dispute.*arbitration.*binding', 'regex', 'mandatory_arbitration', 0.9),
('JAMS Arbitration', 'JAMS.*arbitration|arbitration.*JAMS', 'regex', 'arbitration_provider', 0.85),
('AAA Arbitration', 'American Arbitration Association|AAA.*arbitration', 'regex', 'arbitration_provider', 0.85),
('Class Action Waiver', 'waive.*class action|no class action|class action.*waived', 'regex', 'class_waiver', 0.8),
('Opt-out Provision', 'opt.out.*arbitration|arbitration.*opt.out', 'regex', 'opt_out_clause', 0.75);

-- Record migration
INSERT INTO schema_migrations (version, description, checksum) VALUES 
(1, 'Initial schema with optimized indexes for RAG system', 'a1b2c3d4e5f6');

-- Analyze tables for query optimization
ANALYZE;