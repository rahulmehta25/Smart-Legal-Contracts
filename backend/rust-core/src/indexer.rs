//! Document indexing engine with memory-mapped files and advanced search

use crate::{CoreError, CoreResult, ParsedDocument};
use memmap2::{Mmap, MmapMut, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};

/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub index_path: PathBuf,
    pub memory_mapped: bool,
    pub compression_enabled: bool,
    pub max_index_size: usize,
    pub bloom_filter_size: usize,
    pub enable_fuzzy_search: bool,
    pub min_word_length: usize,
    pub max_word_length: usize,
    pub stop_words: HashSet<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let mut stop_words = HashSet::new();
        stop_words.extend([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their"
        ].iter().map(|s| s.to_string()));

        Self {
            index_path: PathBuf::from("index"),
            memory_mapped: true,
            compression_enabled: true,
            max_index_size: 1024 * 1024 * 1024, // 1GB
            bloom_filter_size: 1_000_000,
            enable_fuzzy_search: true,
            min_word_length: 3,
            max_word_length: 50,
            stop_words,
        }
    }
}

/// Document index structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIndex {
    pub document_id: String,
    pub title: String,
    pub content_hash: String,
    pub word_count: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: IndexMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub file_path: Option<String>,
    pub file_size: usize,
    pub file_type: String,
    pub language: Option<String>,
    pub tags: Vec<String>,
    pub custom_fields: HashMap<String, String>,
}

/// Search result with relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document_id: String,
    pub title: String,
    pub snippet: String,
    pub relevance_score: f64,
    pub matched_terms: Vec<String>,
    pub highlights: Vec<Highlight>,
    pub metadata: IndexMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub field: String,
}

/// Search query structure
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub terms: Vec<String>,
    pub filters: HashMap<String, String>,
    pub fuzzy: bool,
    pub phrase: bool,
    pub boost_fields: HashMap<String, f64>,
    pub limit: usize,
    pub offset: usize,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            filters: HashMap::new(),
            fuzzy: false,
            phrase: false,
            boost_fields: HashMap::new(),
            limit: 10,
            offset: 0,
        }
    }
}

/// High-performance document indexer
pub struct DocumentIndexer {
    config: IndexConfig,
    inverted_index: Arc<RwLock<InvertedIndex>>,
    bloom_filter: Arc<RwLock<BloomFilter>>,
    document_store: Arc<AsyncRwLock<DocumentStore>>,
    memory_mapped_files: Arc<AsyncMutex<HashMap<String, Mmap>>>,
}

/// Inverted index for fast text search
#[derive(Debug)]
struct InvertedIndex {
    // Term -> Vec<(document_id, positions, tf_idf_score)>
    terms: BTreeMap<String, Vec<TermEntry>>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
}

#[derive(Debug, Clone)]
struct TermEntry {
    document_id: String,
    positions: Vec<usize>,
    term_frequency: usize,
    tf_idf_score: f64,
}

/// Bloom filter for fast negative lookups
struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hash_functions: usize,
    hash_seeds: Vec<u64>,
}

/// Document storage with memory mapping
struct DocumentStore {
    documents: HashMap<String, DocumentIndex>,
    file_mappings: HashMap<String, FileMapping>,
    storage_file: Option<File>,
}

#[derive(Debug, Clone)]
struct FileMapping {
    offset: usize,
    length: usize,
    compressed: bool,
}

impl DocumentIndexer {
    /// Create new document indexer
    pub fn new(config: IndexConfig) -> CoreResult<Self> {
        std::fs::create_dir_all(&config.index_path)?;
        
        let inverted_index = Arc::new(RwLock::new(InvertedIndex::new()));
        let bloom_filter = Arc::new(RwLock::new(BloomFilter::new(
            config.bloom_filter_size,
            5, // num hash functions
        )));
        let document_store = Arc::new(AsyncRwLock::new(DocumentStore::new()?));
        let memory_mapped_files = Arc::new(AsyncMutex::new(HashMap::new()));

        Ok(Self {
            config,
            inverted_index,
            bloom_filter,
            document_store,
            memory_mapped_files,
        })
    }

    /// Add document to index
    pub async fn add_document(&self, doc: ParsedDocument) -> CoreResult<()> {
        let document_id = doc.id.clone();
        
        // Create document index entry
        let doc_index = DocumentIndex {
            document_id: document_id.clone(),
            title: doc.structure.sections.first()
                .map(|s| s.title.clone())
                .unwrap_or_else(|| "Untitled".to_string()),
            content_hash: self.calculate_content_hash(&doc.content),
            word_count: doc.statistics.total_words,
            created_at: doc.metadata.created_at.unwrap_or_else(|| chrono::Utc::now()),
            updated_at: chrono::Utc::now(),
            metadata: IndexMetadata {
                file_path: None,
                file_size: doc.metadata.size,
                file_type: format!("{:?}", doc.metadata.file_type),
                language: doc.metadata.language.clone(),
                tags: Vec::new(),
                custom_fields: HashMap::new(),
            },
        };

        // Tokenize and process content
        let tokens = self.tokenize_content(&doc.content);
        
        // Update inverted index
        self.update_inverted_index(&document_id, &tokens).await?;
        
        // Update bloom filter
        {
            let mut bloom = self.bloom_filter.write().unwrap();
            for token in &tokens {
                bloom.insert(&token.text);
            }
        }

        // Store document
        {
            let mut store = self.document_store.write().await;
            store.add_document(doc_index).await?;
        }

        // Memory map document content if enabled
        if self.config.memory_mapped {
            self.create_memory_mapping(&document_id, &doc.content).await?;
        }

        Ok(())
    }

    /// Search documents with relevance scoring
    pub async fn search(&self, query: SearchQuery) -> CoreResult<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        // Quick bloom filter check
        {
            let bloom = self.bloom_filter.read().unwrap();
            let mut has_any_term = false;
            for term in &query.terms {
                if bloom.contains(term) {
                    has_any_term = true;
                    break;
                }
            }
            if !has_any_term && !query.terms.is_empty() {
                return Ok(Vec::new()); // No results possible
            }
        }

        // Search inverted index
        let candidate_docs = self.search_inverted_index(&query).await?;
        
        // Score and rank results
        let mut scored_results = self.score_search_results(candidate_docs, &query).await?;
        
        // Sort by relevance score
        scored_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        
        // Apply pagination
        let start_idx = query.offset;
        let end_idx = std::cmp::min(start_idx + query.limit, scored_results.len());
        let paginated_results = scored_results[start_idx..end_idx].to_vec();
        
        tracing::debug!(
            "Search completed in {:?}ms, found {} results",
            start_time.elapsed().as_millis(),
            paginated_results.len()
        );

        Ok(paginated_results)
    }

    /// Bulk index multiple documents in parallel
    pub async fn bulk_index(&self, documents: Vec<ParsedDocument>) -> CoreResult<Vec<String>> {
        let chunk_size = std::cmp::max(1, documents.len() / rayon::current_num_threads());
        let mut document_ids = Vec::new();

        // Process documents in parallel chunks
        let chunks: Vec<_> = documents.chunks(chunk_size).collect();
        let chunk_results: Vec<Vec<String>> = tokio::task::spawn_blocking(move || {
            chunks
                .par_iter()
                .map(|chunk| {
                    let mut chunk_ids = Vec::new();
                    for doc in *chunk {
                        chunk_ids.push(doc.id.clone());
                        // Note: In real implementation, would need to handle async properly
                    }
                    chunk_ids
                })
                .collect()
        }).await.map_err(|e| CoreError::IndexError(e.to_string()))?;

        for chunk_result in chunk_results {
            document_ids.extend(chunk_result);
        }

        // Process each document (simplified for example)
        for doc in documents {
            self.add_document(doc).await?;
        }

        Ok(document_ids)
    }

    /// Remove document from index
    pub async fn remove_document(&self, document_id: &str) -> CoreResult<()> {
        // Remove from inverted index
        {
            let mut index = self.inverted_index.write().unwrap();
            index.remove_document(document_id);
        }

        // Remove from document store
        {
            let mut store = self.document_store.write().await;
            store.remove_document(document_id).await?;
        }

        // Remove memory mapping
        {
            let mut mappings = self.memory_mapped_files.lock().await;
            mappings.remove(document_id);
        }

        Ok(())
    }

    /// Get document statistics
    pub async fn get_statistics(&self) -> CoreResult<IndexStatistics> {
        let index = self.inverted_index.read().unwrap();
        let store = self.document_store.read().await;
        
        Ok(IndexStatistics {
            total_documents: index.total_documents,
            unique_terms: index.terms.len(),
            total_size: self.calculate_index_size().await?,
            memory_usage: self.calculate_memory_usage().await?,
            last_updated: chrono::Utc::now(),
        })
    }

    /// Optimize index by rebuilding and compacting
    pub async fn optimize(&self) -> CoreResult<()> {
        // Rebuild inverted index with optimizations
        {
            let mut index = self.inverted_index.write().unwrap();
            index.optimize();
        }

        // Compact document store
        {
            let mut store = self.document_store.write().await;
            store.compact().await?;
        }

        // Rebuild bloom filter
        {
            let mut bloom = self.bloom_filter.write().unwrap();
            bloom.clear();
            
            let index = self.inverted_index.read().unwrap();
            for term in index.terms.keys() {
                bloom.insert(term);
            }
        }

        Ok(())
    }

    /// Create memory mapping for document content
    async fn create_memory_mapping(&self, document_id: &str, content: &str) -> CoreResult<()> {
        let file_path = self.config.index_path.join(format!("{}.mmap", document_id));
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&file_path)?;

        file.write_all(content.as_bytes())?;
        file.sync_all()?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| CoreError::IndexError(e.to_string()))?
        };

        let mut mappings = self.memory_mapped_files.lock().await;
        mappings.insert(document_id.to_string(), mmap);

        Ok(())
    }

    /// Tokenize content into searchable terms
    fn tokenize_content(&self, content: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current_pos = 0;

        for word in content.split_whitespace() {
            let cleaned = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>()
                .to_lowercase();

            if cleaned.len() >= self.config.min_word_length
                && cleaned.len() <= self.config.max_word_length
                && !self.config.stop_words.contains(&cleaned)
            {
                tokens.push(Token {
                    text: cleaned,
                    position: current_pos,
                    original: word.to_string(),
                });
            }

            current_pos += word.len() + 1; // +1 for space
        }

        tokens
    }

    /// Calculate content hash for deduplication
    fn calculate_content_hash(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Update inverted index with new document tokens
    async fn update_inverted_index(&self, document_id: &str, tokens: &[Token]) -> CoreResult<()> {
        let mut index = self.inverted_index.write().unwrap();
        
        // Calculate term frequencies
        let mut term_frequencies = HashMap::new();
        for token in tokens {
            let entry = term_frequencies.entry(token.text.clone()).or_insert(Vec::new());
            entry.push(token.position);
        }

        // Update index
        for (term, positions) in term_frequencies {
            let term_frequency = positions.len();
            let tf_idf_score = self.calculate_tf_idf_score(term_frequency, &term, &index);
            
            let term_entry = TermEntry {
                document_id: document_id.to_string(),
                positions,
                term_frequency,
                tf_idf_score,
            };

            index.add_term_entry(term, term_entry);
        }

        index.total_documents += 1;
        Ok(())
    }

    /// Calculate TF-IDF score for relevance ranking
    fn calculate_tf_idf_score(&self, term_frequency: usize, term: &str, index: &InvertedIndex) -> f64 {
        let tf = term_frequency as f64;
        let df = index.document_frequencies.get(term).unwrap_or(&1);
        let idf = (*df as f64).log10();
        
        (tf * idf).max(0.01) // Minimum score
    }

    /// Search inverted index for query terms
    async fn search_inverted_index(&self, query: &SearchQuery) -> CoreResult<Vec<String>> {
        let index = self.inverted_index.read().unwrap();
        let mut candidate_docs = HashSet::new();

        if query.phrase {
            // Phrase search - find documents with terms in sequence
            candidate_docs = self.phrase_search(&query.terms, &index)?;
        } else if query.terms.len() == 1 {
            // Single term search
            if let Some(entries) = index.terms.get(&query.terms[0]) {
                for entry in entries {
                    candidate_docs.insert(entry.document_id.clone());
                }
            }
        } else {
            // Multiple terms - find intersection
            let mut term_results: Vec<HashSet<String>> = Vec::new();
            
            for term in &query.terms {
                let mut term_docs = HashSet::new();
                if let Some(entries) = index.terms.get(term) {
                    for entry in entries {
                        term_docs.insert(entry.document_id.clone());
                    }
                }
                term_results.push(term_docs);
            }

            if !term_results.is_empty() {
                candidate_docs = term_results[0].clone();
                for term_docs in &term_results[1..] {
                    candidate_docs = candidate_docs.intersection(term_docs).cloned().collect();
                }
            }
        }

        Ok(candidate_docs.into_iter().collect())
    }

    /// Phrase search implementation
    fn phrase_search(&self, terms: &[String], index: &InvertedIndex) -> CoreResult<HashSet<String>> {
        let mut result_docs = HashSet::new();
        
        if terms.is_empty() {
            return Ok(result_docs);
        }

        // Get documents containing all terms
        let mut term_doc_positions: Vec<HashMap<String, Vec<usize>>> = Vec::new();
        
        for term in terms {
            let mut term_positions = HashMap::new();
            if let Some(entries) = index.terms.get(term) {
                for entry in entries {
                    term_positions.insert(entry.document_id.clone(), entry.positions.clone());
                }
            }
            term_doc_positions.push(term_positions);
        }

        // Find documents with consecutive positions
        if let Some(first_term_positions) = term_doc_positions.first() {
            for (doc_id, first_positions) in first_term_positions {
                let mut has_phrase = false;
                
                'position_loop: for &first_pos in first_positions {
                    let mut current_pos = first_pos;
                    let mut found_sequence = true;
                    
                    // Check if subsequent terms appear at consecutive positions
                    for term_positions in &term_doc_positions[1..] {
                        if let Some(positions) = term_positions.get(doc_id) {
                            current_pos += 1;
                            if !positions.contains(&current_pos) {
                                found_sequence = false;
                                break;
                            }
                        } else {
                            found_sequence = false;
                            break;
                        }
                    }
                    
                    if found_sequence {
                        has_phrase = true;
                        break 'position_loop;
                    }
                }
                
                if has_phrase {
                    result_docs.insert(doc_id.clone());
                }
            }
        }

        Ok(result_docs)
    }

    /// Score search results for relevance ranking
    async fn score_search_results(&self, candidate_docs: Vec<String>, query: &SearchQuery) -> CoreResult<Vec<SearchResult>> {
        let mut results = Vec::new();
        let store = self.document_store.read().await;
        let index = self.inverted_index.read().unwrap();

        for doc_id in candidate_docs {
            if let Some(doc_index) = store.documents.get(&doc_id) {
                let mut relevance_score = 0.0;
                let mut matched_terms = Vec::new();

                // Calculate relevance based on TF-IDF scores
                for term in &query.terms {
                    if let Some(entries) = index.terms.get(term) {
                        for entry in entries {
                            if entry.document_id == doc_id {
                                relevance_score += entry.tf_idf_score;
                                matched_terms.push(term.clone());
                                break;
                            }
                        }
                    }
                }

                // Apply field boosts
                for (field, boost) in &query.boost_fields {
                    if field == "title" && doc_index.title.to_lowercase().contains(&query.terms.join(" ").to_lowercase()) {
                        relevance_score *= boost;
                    }
                }

                // Generate snippet
                let snippet = self.generate_snippet(&doc_id, &query.terms).await?;
                
                results.push(SearchResult {
                    document_id: doc_id,
                    title: doc_index.title.clone(),
                    snippet,
                    relevance_score,
                    matched_terms,
                    highlights: Vec::new(), // Would implement highlighting
                    metadata: doc_index.metadata.clone(),
                });
            }
        }

        Ok(results)
    }

    /// Generate search result snippet
    async fn generate_snippet(&self, document_id: &str, terms: &[String]) -> CoreResult<String> {
        let mappings = self.memory_mapped_files.lock().await;
        
        if let Some(mmap) = mappings.get(document_id) {
            let content = std::str::from_utf8(mmap)
                .map_err(|e| CoreError::IndexError(e.to_string()))?;
            
            // Find first occurrence of any term
            let mut best_start = 0;
            for term in terms {
                if let Some(pos) = content.to_lowercase().find(&term.to_lowercase()) {
                    best_start = pos;
                    break;
                }
            }
            
            // Extract snippet around the term
            let snippet_start = best_start.saturating_sub(100);
            let snippet_end = std::cmp::min(best_start + 200, content.len());
            
            Ok(content[snippet_start..snippet_end].to_string())
        } else {
            Ok("No snippet available".to_string())
        }
    }

    /// Calculate total index size
    async fn calculate_index_size(&self) -> CoreResult<usize> {
        let mut total_size = 0;
        
        if let Ok(entries) = std::fs::read_dir(&self.config.index_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len() as usize;
                    }
                }
            }
        }
        
        Ok(total_size)
    }

    /// Calculate memory usage
    async fn calculate_memory_usage(&self) -> CoreResult<usize> {
        // Simplified memory usage calculation
        let mappings = self.memory_mapped_files.lock().await;
        let memory_usage = mappings.values().map(|mmap| mmap.len()).sum();
        Ok(memory_usage)
    }
}

/// Token structure for indexing
#[derive(Debug, Clone)]
struct Token {
    text: String,
    position: usize,
    original: String,
}

/// Index statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub total_documents: usize,
    pub unique_terms: usize,
    pub total_size: usize,
    pub memory_usage: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl InvertedIndex {
    fn new() -> Self {
        Self {
            terms: BTreeMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
        }
    }

    fn add_term_entry(&mut self, term: String, entry: TermEntry) {
        let entries = self.terms.entry(term.clone()).or_insert_with(Vec::new);
        entries.push(entry);
        
        *self.document_frequencies.entry(term).or_insert(0) += 1;
    }

    fn remove_document(&mut self, document_id: &str) {
        self.terms.retain(|_term, entries| {
            entries.retain(|entry| entry.document_id != document_id);
            !entries.is_empty()
        });
        
        self.total_documents = self.total_documents.saturating_sub(1);
    }

    fn optimize(&mut self) {
        // Remove empty entries and rebuild document frequencies
        self.terms.retain(|_term, entries| !entries.is_empty());
        
        self.document_frequencies.clear();
        for (term, entries) in &self.terms {
            self.document_frequencies.insert(term.clone(), entries.len());
        }
    }
}

impl BloomFilter {
    fn new(num_bits: usize, num_hash_functions: usize) -> Self {
        let bits = vec![0u64; (num_bits + 63) / 64];
        let hash_seeds = (0..num_hash_functions).map(|i| i as u64 * 0x9e3779b9).collect();
        
        Self {
            bits,
            num_bits,
            num_hash_functions,
            hash_seeds,
        }
    }

    fn insert(&mut self, item: &str) {
        for &seed in &self.hash_seeds {
            let hash = self.hash_with_seed(item, seed);
            let bit_index = (hash % self.num_bits as u64) as usize;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            
            self.bits[word_index] |= 1u64 << bit_offset;
        }
    }

    fn contains(&self, item: &str) -> bool {
        for &seed in &self.hash_seeds {
            let hash = self.hash_with_seed(item, seed);
            let bit_index = (hash % self.num_bits as u64) as usize;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            
            if (self.bits[word_index] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    fn clear(&mut self) {
        self.bits.fill(0);
    }

    fn hash_with_seed(&self, item: &str, seed: u64) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }
}

impl DocumentStore {
    fn new() -> CoreResult<Self> {
        Ok(Self {
            documents: HashMap::new(),
            file_mappings: HashMap::new(),
            storage_file: None,
        })
    }

    async fn add_document(&mut self, doc: DocumentIndex) -> CoreResult<()> {
        self.documents.insert(doc.document_id.clone(), doc);
        Ok(())
    }

    async fn remove_document(&mut self, document_id: &str) -> CoreResult<()> {
        self.documents.remove(document_id);
        self.file_mappings.remove(document_id);
        Ok(())
    }

    async fn compact(&mut self) -> CoreResult<()> {
        // Implement storage compaction
        self.file_mappings.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_indexer_creation() {
        let temp_dir = tempdir().unwrap();
        let config = IndexConfig {
            index_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let indexer = DocumentIndexer::new(config).unwrap();
        assert!(true);
    }

    #[tokio::test]
    async fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 3);
        
        bloom.insert("hello");
        bloom.insert("world");
        
        assert!(bloom.contains("hello"));
        assert!(bloom.contains("world"));
        assert!(!bloom.contains("nonexistent"));
    }

    #[test]
    fn test_tokenization() {
        let config = IndexConfig::default();
        let indexer = DocumentIndexer::new(config).unwrap();
        
        let tokens = indexer.tokenize_content("Hello world! This is a test.");
        
        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|t| t.text == "hello"));
        assert!(tokens.iter().any(|t| t.text == "world"));
        assert!(tokens.iter().any(|t| t.text == "test"));
    }
}