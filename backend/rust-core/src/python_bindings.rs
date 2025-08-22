//! Python bindings with PyO3 for seamless integration

use crate::{
    CoreError, CoreResult, DocumentParser, PatternMatcher, DocumentIndexer,
    ParsedDocument, Pattern, PatternMatch, SearchQuery, SearchResult,
    CryptoEngine, EncryptedData, HashAlgorithm, EncryptionAlgorithm,
    LockFreeCache, HighPerformanceCache, CacheConfig,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{exceptions::PyException, wrap_pyfunction};
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// Python-compatible document parser
#[pyclass]
pub struct PyDocumentParser {
    parser: DocumentParser,
    config: crate::parser::ParserConfig,
}

/// Python-compatible pattern matcher
#[pyclass]
pub struct PyPatternMatcher {
    matcher: Arc<AsyncMutex<PatternMatcher>>,
    patterns: Vec<Pattern>,
}

/// Python-compatible document indexer
#[pyclass]
pub struct PyDocumentIndexer {
    indexer: Arc<AsyncMutex<DocumentIndexer>>,
}

/// Python-compatible crypto engine
#[pyclass]
pub struct PyCryptoEngine {
    engine: CryptoEngine,
}

/// Python-compatible cache
#[pyclass]
pub struct PyCache {
    cache: Arc<LockFreeCache<String, Vec<u8>>>,
}

/// Python-compatible parsed document
#[pyclass]
#[derive(Clone)]
pub struct PyParsedDocument {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub file_type: String,
    #[pyo3(get)]
    pub language: Option<String>,
}

/// Python-compatible pattern match
#[pyclass]
#[derive(Clone)]
pub struct PyPatternMatch {
    #[pyo3(get)]
    pub pattern: String,
    #[pyo3(get)]
    pub matched_text: String,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub match_type: String,
}

/// Python-compatible search result
#[pyclass]
#[derive(Clone)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub document_id: String,
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub snippet: String,
    #[pyo3(get)]
    pub relevance_score: f64,
    #[pyo3(get)]
    pub matched_terms: Vec<String>,
}

/// Python-compatible encrypted data
#[pyclass]
#[derive(Clone)]
pub struct PyEncryptedData {
    #[pyo3(get)]
    pub algorithm: String,
    #[pyo3(get)]
    pub ciphertext: Vec<u8>,
    #[pyo3(get)]
    pub nonce: Vec<u8>,
}

#[pymethods]
impl PyDocumentParser {
    /// Create new document parser
    #[new]
    pub fn new(
        enable_parallel: Option<bool>,
        chunk_size: Option<usize>,
        memory_mapped: Option<bool>,
    ) -> Self {
        let config = crate::parser::ParserConfig {
            enable_parallel: enable_parallel.unwrap_or(true),
            chunk_size: chunk_size.unwrap_or(64 * 1024),
            memory_mapped: memory_mapped.unwrap_or(true),
            ..Default::default()
        };
        
        Self {
            parser: DocumentParser::new(config.clone()),
            config,
        }
    }

    /// Parse document from file path
    pub fn parse_file(&self, path: &str) -> PyResult<PyParsedDocument> {
        let doc = self.parser.parse_file(path)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        Ok(PyParsedDocument::from_parsed_document(doc))
    }

    /// Parse document from text content
    pub fn parse_text(&self, text: &str, file_type: Option<&str>) -> PyResult<PyParsedDocument> {
        // Create temporary file for parsing
        let temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        std::fs::write(temp_file.path(), text)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        let doc = self.parser.parse_file(temp_file.path())
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        Ok(PyParsedDocument::from_parsed_document(doc))
    }

    /// Parse document from bytes
    pub fn parse_bytes(&self, data: &[u8]) -> PyResult<PyParsedDocument> {
        let text = String::from_utf8(data.to_vec())
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        self.parse_text(&text, None)
    }

    /// Get parser statistics
    pub fn get_stats(&self) -> PyResult<PyDict> {
        let py = Python::acquire_gil().python();
        let dict = PyDict::new(py);
        
        dict.set_item("config", format!("{:?}", self.config))?;
        dict.set_item("memory_usage", "N/A")?; // Would implement actual memory tracking
        
        Ok(dict.to_object(py).downcast::<PyDict>(py).unwrap().clone())
    }
}

#[pymethods]
impl PyPatternMatcher {
    /// Create new pattern matcher
    #[new]
    pub fn new(
        case_sensitive: Option<bool>,
        enable_fuzzy: Option<bool>,
        max_edit_distance: Option<usize>,
    ) -> Self {
        let config = crate::pattern_matcher::PatternConfig {
            case_sensitive: case_sensitive.unwrap_or(false),
            enable_fuzzy: enable_fuzzy.unwrap_or(true),
            max_edit_distance: max_edit_distance.unwrap_or(2),
            ..Default::default()
        };
        
        Self {
            matcher: Arc::new(AsyncMutex::new(PatternMatcher::new(config))),
            patterns: Vec::new(),
        }
    }

    /// Add literal pattern
    pub fn add_literal_pattern(&mut self, pattern: &str) -> PyResult<()> {
        self.patterns.push(Pattern::Literal(pattern.to_string()));
        Ok(())
    }

    /// Add regex pattern
    pub fn add_regex_pattern(&mut self, pattern: &str) -> PyResult<()> {
        self.patterns.push(Pattern::Regex(pattern.to_string()));
        Ok(())
    }

    /// Add fuzzy pattern
    pub fn add_fuzzy_pattern(&mut self, pattern: &str, max_distance: Option<usize>) -> PyResult<()> {
        self.patterns.push(Pattern::Fuzzy {
            text: pattern.to_string(),
            max_distance: max_distance.unwrap_or(2),
        });
        Ok(())
    }

    /// Compile patterns for efficient matching
    pub fn compile_patterns(&mut self) -> PyResult<()> {
        let matcher = self.matcher.clone();
        let patterns = self.patterns.clone();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let mut matcher = matcher.lock().await;
            matcher.compile_patterns(patterns)
                .map_err(|e| PyException::new_err(e.to_string()))
        })
    }

    /// Find matches in text
    pub fn find_matches(&self, text: &str) -> PyResult<Vec<PyPatternMatch>> {
        let matcher = self.matcher.clone();
        let text = text.to_string();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let matcher = matcher.lock().await;
            let matches = matcher.find_matches(&text)
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            Ok(matches.into_iter().map(PyPatternMatch::from_pattern_match).collect())
        })
    }

    /// Find matches with parallel processing
    pub fn find_matches_parallel(&self, text: &str, chunk_size: Option<usize>) -> PyResult<Vec<PyPatternMatch>> {
        let matcher = self.matcher.clone();
        let text = text.to_string();
        let chunk_size = chunk_size.unwrap_or(1000);
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let matcher = matcher.lock().await;
            let matches = matcher.find_matches_parallel(&text, chunk_size)
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            Ok(matches.into_iter().map(PyPatternMatch::from_pattern_match).collect())
        })
    }

    /// Analyze patterns in text
    pub fn analyze_patterns(&self, text: &str) -> PyResult<PyDict> {
        let matcher = self.matcher.clone();
        let text = text.to_string();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let matcher = matcher.lock().await;
            let analysis = matcher.analyze_patterns(&text)
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            let py = Python::acquire_gil().python();
            let dict = PyDict::new(py);
            
            dict.set_item("total_matches", analysis.total_matches)?;
            dict.set_item("unique_patterns", analysis.unique_patterns)?;
            dict.set_item("average_confidence", analysis.average_confidence)?;
            dict.set_item("coverage_percentage", analysis.coverage_percentage)?;
            
            Ok(dict.to_object(py).downcast::<PyDict>(py).unwrap().clone())
        })
    }

    /// Get pattern statistics
    pub fn get_pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[pymethods]
impl PyDocumentIndexer {
    /// Create new document indexer
    #[new]
    pub fn new(
        index_path: Option<&str>,
        max_capacity: Option<usize>,
        enable_caching: Option<bool>,
    ) -> PyResult<Self> {
        let config = crate::indexer::IndexConfig {
            index_path: std::path::PathBuf::from(index_path.unwrap_or("index")),
            max_index_size: max_capacity.unwrap_or(1024 * 1024 * 1024), // 1GB
            enable_fuzzy_search: enable_caching.unwrap_or(true),
            ..Default::default()
        };
        
        let indexer = DocumentIndexer::new(config)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        Ok(Self {
            indexer: Arc::new(AsyncMutex::new(indexer)),
        })
    }

    /// Add document to index
    pub fn add_document(&self, doc: PyParsedDocument) -> PyResult<()> {
        let indexer = self.indexer.clone();
        let parsed_doc = doc.to_parsed_document();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let indexer = indexer.lock().await;
            indexer.add_document(parsed_doc).await
                .map_err(|e| PyException::new_err(e.to_string()))
        })
    }

    /// Search documents
    pub fn search(&self, query: &str, limit: Option<usize>) -> PyResult<Vec<PySearchResult>> {
        let indexer = self.indexer.clone();
        let search_query = SearchQuery {
            terms: query.split_whitespace().map(|s| s.to_string()).collect(),
            limit: limit.unwrap_or(10),
            ..Default::default()
        };
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let indexer = indexer.lock().await;
            let results = indexer.search(search_query).await
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            Ok(results.into_iter().map(PySearchResult::from_search_result).collect())
        })
    }

    /// Bulk index documents
    pub fn bulk_index(&self, documents: Vec<PyParsedDocument>) -> PyResult<Vec<String>> {
        let indexer = self.indexer.clone();
        let parsed_docs: Vec<ParsedDocument> = documents
            .into_iter()
            .map(|d| d.to_parsed_document())
            .collect();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let indexer = indexer.lock().await;
            indexer.bulk_index(parsed_docs).await
                .map_err(|e| PyException::new_err(e.to_string()))
        })
    }

    /// Remove document from index
    pub fn remove_document(&self, document_id: &str) -> PyResult<()> {
        let indexer = self.indexer.clone();
        let doc_id = document_id.to_string();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let indexer = indexer.lock().await;
            indexer.remove_document(&doc_id).await
                .map_err(|e| PyException::new_err(e.to_string()))
        })
    }

    /// Get index statistics
    pub fn get_statistics(&self) -> PyResult<PyDict> {
        let indexer = self.indexer.clone();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let indexer = indexer.lock().await;
            let stats = indexer.get_statistics().await
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            let py = Python::acquire_gil().python();
            let dict = PyDict::new(py);
            
            dict.set_item("total_documents", stats.total_documents)?;
            dict.set_item("unique_terms", stats.unique_terms)?;
            dict.set_item("total_size", stats.total_size)?;
            dict.set_item("memory_usage", stats.memory_usage)?;
            
            Ok(dict.to_object(py).downcast::<PyDict>(py).unwrap().clone())
        })
    }
}

#[pymethods]
impl PyCryptoEngine {
    /// Create new crypto engine
    #[new]
    pub fn new(
        algorithm: Option<&str>,
        key_derivation_rounds: Option<u32>,
        enable_compression: Option<bool>,
    ) -> Self {
        let config = crate::crypto::CryptoConfig {
            default_algorithm: match algorithm.unwrap_or("AES256GCM") {
                "ChaCha20Poly1305" => EncryptionAlgorithm::ChaCha20Poly1305,
                "AES128GCM" => EncryptionAlgorithm::AES128GCM,
                _ => EncryptionAlgorithm::AES256GCM,
            },
            key_derivation_rounds: key_derivation_rounds.unwrap_or(100_000),
            enable_compression: enable_compression.unwrap_or(false),
            ..Default::default()
        };
        
        Self {
            engine: CryptoEngine::new(config),
        }
    }

    /// Encrypt data with password
    pub fn encrypt_with_password(&self, data: &[u8], password: &str) -> PyResult<PyEncryptedData> {
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            let encrypted = self.engine.encrypt_with_password(data, password).await
                .map_err(|e| PyException::new_err(e.to_string()))?;
            
            Ok(PyEncryptedData::from_encrypted_data(encrypted))
        })
    }

    /// Decrypt data with password
    pub fn decrypt_with_password(&self, encrypted_data: PyEncryptedData, password: &str) -> PyResult<Vec<u8>> {
        let encrypted = encrypted_data.to_encrypted_data();
        
        pyo3_asyncio::tokio::get_runtime().block_on(async move {
            self.engine.decrypt_with_password(&encrypted, password).await
                .map_err(|e| PyException::new_err(e.to_string()))
        })
    }

    /// Generate hash
    pub fn hash(&self, data: &[u8], algorithm: Option<&str>) -> PyResult<Vec<u8>> {
        let hash_algo = match algorithm.unwrap_or("SHA256") {
            "SHA384" => HashAlgorithm::SHA384,
            "SHA512" => HashAlgorithm::SHA512,
            "Blake3" => HashAlgorithm::Blake3,
            "Argon2id" => HashAlgorithm::Argon2id,
            _ => HashAlgorithm::SHA256,
        };
        
        self.engine.hash(data, hash_algo)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Verify hash
    pub fn verify_hash(&self, data: &[u8], hash: &[u8], algorithm: Option<&str>) -> PyResult<bool> {
        let hash_algo = match algorithm.unwrap_or("SHA256") {
            "SHA384" => HashAlgorithm::SHA384,
            "SHA512" => HashAlgorithm::SHA512,
            "Blake3" => HashAlgorithm::Blake3,
            "Argon2id" => HashAlgorithm::Argon2id,
            _ => HashAlgorithm::SHA256,
        };
        
        self.engine.verify_hash(data, hash, hash_algo)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Generate random bytes
    pub fn generate_random_bytes(&self, length: usize) -> PyResult<Vec<u8>> {
        self.engine.generate_random_bytes(length)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Derive key from password
    pub fn derive_key(&self, password: &str, salt: &[u8], rounds: Option<u32>) -> PyResult<Vec<u8>> {
        let derived_key = self.engine.derive_key_from_password(
            password,
            salt,
            rounds.unwrap_or(100_000)
        ).map_err(|e| PyException::new_err(e.to_string()))?;
        
        Ok(derived_key.key)
    }
}

#[pymethods]
impl PyCache {
    /// Create new cache
    #[new]
    pub fn new(max_capacity: Option<usize>, max_memory: Option<usize>) -> Self {
        let config = CacheConfig {
            max_capacity: max_capacity.unwrap_or(10_000),
            max_memory: max_memory.unwrap_or(100 * 1024 * 1024), // 100MB
            ..Default::default()
        };
        
        Self {
            cache: Arc::new(LockFreeCache::new(config)),
        }
    }

    /// Get value from cache
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.cache.get(&key.to_string())
    }

    /// Put value in cache
    pub fn put(&self, key: &str, value: &[u8]) -> PyResult<()> {
        self.cache.put(key.to_string(), value.to_vec())
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Remove value from cache
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        self.cache.remove(&key.to_string())
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.cache.contains_key(&key.to_string())
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.size()
    }

    /// Get cache statistics
    pub fn stats(&self) -> PyResult<PyDict> {
        let stats = self.cache.stats();
        let py = Python::acquire_gil().python();
        let dict = PyDict::new(py);
        
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("evictions", stats.evictions)?;
        dict.set_item("size", stats.size)?;
        dict.set_item("memory_usage", stats.memory_usage)?;
        dict.set_item("hit_rate", stats.hit_rate)?;
        
        Ok(dict.to_object(py).downcast::<PyDict>(py).unwrap().clone())
    }
}

// Helper implementations for conversion between Rust and Python types
impl PyParsedDocument {
    fn from_parsed_document(doc: ParsedDocument) -> Self {
        Self {
            id: doc.id,
            content: doc.content,
            word_count: doc.statistics.total_words,
            file_type: format!("{:?}", doc.metadata.file_type),
            language: doc.metadata.language,
        }
    }

    fn to_parsed_document(&self) -> ParsedDocument {
        ParsedDocument {
            id: self.id.clone(),
            content: self.content.clone(),
            metadata: crate::parser::DocumentMetadata {
                file_type: crate::parser::FileType::PlainText, // Simplified
                size: self.content.len(),
                encoding: "UTF-8".to_string(),
                language: self.language.clone(),
                created_at: Some(chrono::Utc::now()),
            },
            structure: crate::parser::DocumentStructure {
                paragraphs: Vec::new(),
                sections: Vec::new(),
                tables: Vec::new(),
                links: Vec::new(),
            },
            statistics: crate::parser::ParseStatistics {
                total_chars: self.content.len(),
                total_words: self.word_count,
                total_lines: self.content.lines().count(),
                parse_time_ms: 0,
                memory_used: self.content.len(),
            },
        }
    }
}

impl PyPatternMatch {
    fn from_pattern_match(match_result: PatternMatch) -> Self {
        Self {
            pattern: match_result.pattern,
            matched_text: match_result.matched_text,
            start: match_result.start,
            end: match_result.end,
            confidence: match_result.confidence,
            match_type: format!("{:?}", match_result.match_type),
        }
    }
}

impl PySearchResult {
    fn from_search_result(result: SearchResult) -> Self {
        Self {
            document_id: result.document_id,
            title: result.title,
            snippet: result.snippet,
            relevance_score: result.relevance_score,
            matched_terms: result.matched_terms,
        }
    }
}

impl PyEncryptedData {
    fn from_encrypted_data(data: EncryptedData) -> Self {
        Self {
            algorithm: format!("{:?}", data.algorithm),
            ciphertext: data.ciphertext,
            nonce: data.nonce,
        }
    }

    fn to_encrypted_data(&self) -> EncryptedData {
        let algorithm = match self.algorithm.as_str() {
            "ChaCha20Poly1305" => EncryptionAlgorithm::ChaCha20Poly1305,
            "AES128GCM" => EncryptionAlgorithm::AES128GCM,
            _ => EncryptionAlgorithm::AES256GCM,
        };

        EncryptedData {
            algorithm,
            ciphertext: self.ciphertext.clone(),
            nonce: self.nonce.clone(),
            salt: None, // Simplified
            metadata: crate::crypto::EncryptionMetadata {
                created_at: 0,
                key_derivation_rounds: 100_000,
                compressed: false,
                version: 1,
            },
        }
    }
}

// Utility functions for NumPy integration
/// Process NumPy array with pattern matching
#[pyfunction]
pub fn process_numpy_array(py: Python, array: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let array = array.as_array();
    let processed: Vec<f64> = array.iter().map(|&x| x * 2.0).collect();
    Ok(processed.into_pyarray(py).to_owned())
}

/// Batch process multiple texts
#[pyfunction]
pub fn batch_process_texts(
    texts: Vec<String>,
    patterns: Vec<String>,
    parallel: Option<bool>,
) -> PyResult<Vec<Vec<PyPatternMatch>>> {
    let enable_parallel = parallel.unwrap_or(true);
    let mut matcher = PatternMatcher::new(crate::pattern_matcher::PatternConfig {
        enable_parallel,
        ..Default::default()
    });

    let pattern_objects: Vec<Pattern> = patterns
        .into_iter()
        .map(|p| Pattern::Literal(p))
        .collect();

    matcher.compile_patterns(pattern_objects)
        .map_err(|e| PyException::new_err(e.to_string()))?;

    let mut results = Vec::new();
    for text in texts {
        let matches = matcher.find_matches(&text)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        let py_matches: Vec<PyPatternMatch> = matches
            .into_iter()
            .map(PyPatternMatch::from_pattern_match)
            .collect();
        
        results.push(py_matches);
    }

    Ok(results)
}

/// Async function example
#[pyfunction]
pub fn async_process_document(py: Python, text: String) -> PyResult<&PyAny> {
    future_into_py(py, async move {
        // Simulate async processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let config = crate::parser::ParserConfig::default();
        let parser = DocumentParser::new(config);
        
        // Create temporary file
        let temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        std::fs::write(temp_file.path(), text)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        let doc = parser.parse_file(temp_file.path())
            .map_err(|e| PyException::new_err(e.to_string()))?;
        
        Ok(PyParsedDocument::from_parsed_document(doc))
    })
}

/// Memory-efficient text processor
#[pyfunction]
pub fn memory_efficient_process(
    text: &str,
    chunk_size: Option<usize>,
    max_memory: Option<usize>,
) -> PyResult<PyDict> {
    let chunk_size = chunk_size.unwrap_or(64 * 1024);
    let max_memory = max_memory.unwrap_or(100 * 1024 * 1024);
    
    let chunks: Vec<&str> = text
        .as_bytes()
        .chunks(chunk_size)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .collect();
    
    let mut total_words = 0;
    let mut total_lines = 0;
    let mut memory_used = 0;
    
    for chunk in &chunks {
        total_words += chunk.split_whitespace().count();
        total_lines += chunk.lines().count();
        memory_used += chunk.len();
        
        if memory_used > max_memory {
            return Err(PyException::new_err("Memory limit exceeded"));
        }
    }
    
    let py = Python::acquire_gil().python();
    let dict = PyDict::new(py);
    
    dict.set_item("total_chunks", chunks.len())?;
    dict.set_item("total_words", total_words)?;
    dict.set_item("total_lines", total_lines)?;
    dict.set_item("memory_used", memory_used)?;
    dict.set_item("chunk_size", chunk_size)?;
    
    Ok(dict.to_object(py).downcast::<PyDict>(py).unwrap().clone())
}

/// Python module definition
#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyDocumentParser>()?;
    m.add_class::<PyPatternMatcher>()?;
    m.add_class::<PyDocumentIndexer>()?;
    m.add_class::<PyCryptoEngine>()?;
    m.add_class::<PyCache>()?;
    m.add_class::<PyParsedDocument>()?;
    m.add_class::<PyPatternMatch>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyEncryptedData>()?;
    
    // Functions
    m.add_function(wrap_pyfunction!(process_numpy_array, m)?)?;
    m.add_function(wrap_pyfunction!(batch_process_texts, m)?)?;
    m.add_function(wrap_pyfunction!(async_process_document, m)?)?;
    m.add_function(wrap_pyfunction!(memory_efficient_process, m)?)?;
    
    // Constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_CHUNK_SIZE", 64 * 1024)?;
    m.add("DEFAULT_MAX_CAPACITY", 10_000)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_py_document_parser() {
        let parser = PyDocumentParser::new(Some(true), Some(1024), Some(false));
        
        let result = parser.parse_text("Hello, world! This is a test document.", None);
        assert!(result.is_ok());
        
        let doc = result.unwrap();
        assert!(!doc.id.is_empty());
        assert_eq!(doc.content, "Hello, world! This is a test document.");
        assert!(doc.word_count > 0);
    }
    
    #[test]
    fn test_py_pattern_matcher() {
        let mut matcher = PyPatternMatcher::new(Some(false), Some(true), Some(2));
        
        matcher.add_literal_pattern("test").unwrap();
        matcher.add_regex_pattern(r"\b\w+\b").unwrap();
        
        assert_eq!(matcher.get_pattern_count(), 2);
    }
    
    #[test]
    fn test_py_crypto_engine() {
        let engine = PyCryptoEngine::new(Some("AES256GCM"), Some(1000), Some(false));
        
        let data = b"test data";
        let password = "test_password";
        
        let encrypted = engine.encrypt_with_password(data, password).unwrap();
        let decrypted = engine.decrypt_with_password(encrypted, password).unwrap();
        
        assert_eq!(data, decrypted.as_slice());
    }
    
    #[test]
    fn test_py_cache() {
        let cache = PyCache::new(Some(100), Some(1024));
        
        let key = "test_key";
        let value = b"test_value";
        
        cache.put(key, value).unwrap();
        assert!(cache.contains_key(key));
        assert_eq!(cache.get(key), Some(value.to_vec()));
        assert_eq!(cache.size(), 1);
        
        cache.clear();
        assert_eq!(cache.size(), 0);
    }
}