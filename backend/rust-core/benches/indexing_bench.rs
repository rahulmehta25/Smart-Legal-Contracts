//! Document indexing performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_core::{DocumentIndexer, IndexConfig, SearchQuery, ParsedDocument};
use std::collections::HashSet;
use tempfile::tempdir;
use tokio::runtime::Runtime;

/// Benchmark document indexing performance
fn bench_document_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_indexing");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    
    let document_counts = [10, 100, 1000];
    let document_sizes = [1024, 10_240, 102_400]; // 1KB, 10KB, 100KB
    
    for &doc_count in &document_counts {
        for &doc_size in &document_sizes {
            let documents = generate_test_documents(doc_count, doc_size);
            let total_size = doc_count * doc_size;
            
            group.throughput(Throughput::Bytes(total_size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}docs", doc_count), doc_size),
                &documents,
                |b, documents| {
                    b.to_async(&rt).iter(|| async {
                        let config = IndexConfig {
                            index_path: temp_dir.path().join(format!("index_{}", doc_count)),
                            ..Default::default()
                        };
                        
                        let indexer = DocumentIndexer::new(config).unwrap();
                        
                        for doc in black_box(documents) {
                            let _ = indexer.add_document(doc.clone()).await;
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark bulk indexing vs individual indexing
fn bench_bulk_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_indexing");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    let documents = generate_test_documents(500, 5000); // 500 documents, 5KB each
    
    group.throughput(Throughput::Bytes(500 * 5000));
    
    // Individual indexing
    group.bench_function("individual", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("individual_index"),
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            
            for doc in black_box(&documents) {
                let _ = indexer.add_document(doc.clone()).await;
            }
        });
    });
    
    // Bulk indexing
    group.bench_function("bulk", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("bulk_index"),
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            let _ = indexer.bulk_index(black_box(documents.clone())).await;
        });
    });
    
    group.finish();
}

/// Benchmark search performance
fn bench_search_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_performance");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    
    // Pre-populate index
    let documents = generate_test_documents(1000, 10_000); // 1000 docs, 10KB each
    let config = IndexConfig {
        index_path: temp_dir.path().join("search_index"),
        ..Default::default()
    };
    
    let indexer = rt.block_on(async {
        let indexer = DocumentIndexer::new(config).unwrap();
        let _ = indexer.bulk_index(documents).await;
        indexer
    });
    
    let search_terms = [
        vec!["arbitration".to_string()],
        vec!["arbitration".to_string(), "dispute".to_string()],
        vec!["arbitration".to_string(), "dispute".to_string(), "resolution".to_string()],
        vec!["contract".to_string(), "terms".to_string(), "conditions".to_string(), "legal".to_string()],
    ];
    
    for (i, terms) in search_terms.iter().enumerate() {
        let term_count = terms.len();
        
        group.bench_with_input(
            BenchmarkId::new("search", format!("{}terms", term_count)),
            terms,
            |b, terms| {
                b.to_async(&rt).iter(|| async {
                    let query = SearchQuery {
                        terms: black_box(terms.clone()),
                        limit: 50,
                        ..Default::default()
                    };
                    
                    let _ = indexer.search(query).await;
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory-mapped vs regular file I/O
fn bench_memory_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_mapping");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    let documents = generate_test_documents(100, 50_000); // 100 docs, 50KB each
    
    group.throughput(Throughput::Bytes(100 * 50_000));
    
    // Regular file I/O
    group.bench_function("regular_io", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("regular_index"),
                memory_mapped: false,
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            let _ = indexer.bulk_index(black_box(documents.clone())).await;
        });
    });
    
    // Memory-mapped I/O
    group.bench_function("memory_mapped", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("mmap_index"),
                memory_mapped: true,
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            let _ = indexer.bulk_index(black_box(documents.clone())).await;
        });
    });
    
    group.finish();
}

/// Benchmark compression impact
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    let documents = generate_repetitive_documents(200, 20_000); // Highly compressible
    
    group.throughput(Throughput::Bytes(200 * 20_000));
    
    // Without compression
    group.bench_function("no_compression", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("no_comp_index"),
                compression_enabled: false,
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            let _ = indexer.bulk_index(black_box(documents.clone())).await;
        });
    });
    
    // With compression
    group.bench_function("with_compression", |b| {
        b.to_async(&rt).iter(|| async {
            let config = IndexConfig {
                index_path: temp_dir.path().join("comp_index"),
                compression_enabled: true,
                ..Default::default()
            };
            
            let indexer = DocumentIndexer::new(config).unwrap();
            let _ = indexer.bulk_index(black_box(documents.clone())).await;
        });
    });
    
    group.finish();
}

/// Benchmark fuzzy search performance
fn bench_fuzzy_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuzzy_search");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    
    // Pre-populate index with typos and variations
    let documents = generate_documents_with_typos(500, 8000);
    let config = IndexConfig {
        index_path: temp_dir.path().join("fuzzy_index"),
        enable_fuzzy_search: true,
        ..Default::default()
    };
    
    let indexer = rt.block_on(async {
        let indexer = DocumentIndexer::new(config).unwrap();
        let _ = indexer.bulk_index(documents).await;
        indexer
    });
    
    // Exact search
    group.bench_function("exact_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = SearchQuery {
                terms: vec!["arbitration".to_string()],
                fuzzy: false,
                limit: 50,
                ..Default::default()
            };
            
            let _ = indexer.search(black_box(query)).await;
        });
    });
    
    // Fuzzy search
    group.bench_function("fuzzy_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = SearchQuery {
                terms: vec!["arbitraton".to_string()], // Typo
                fuzzy: true,
                limit: 50,
                ..Default::default()
            };
            
            let _ = indexer.search(black_box(query)).await;
        });
    });
    
    group.finish();
}

/// Benchmark index optimization
fn bench_index_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_optimization");
    
    let rt = Runtime::new().unwrap();
    let temp_dir = tempdir().unwrap();
    
    // Create fragmented index
    let documents = generate_test_documents(1000, 5000);
    let config = IndexConfig {
        index_path: temp_dir.path().join("optimize_index"),
        ..Default::default()
    };
    
    let indexer = rt.block_on(async {
        let indexer = DocumentIndexer::new(config).unwrap();
        
        // Add documents individually to create fragmentation
        for doc in &documents {
            let _ = indexer.add_document(doc.clone()).await;
        }
        
        // Remove some documents to create gaps
        for i in (0..documents.len()).step_by(3) {
            let _ = indexer.remove_document(&documents[i].id).await;
        }
        
        indexer
    });
    
    group.bench_function("optimize_index", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = indexer.optimize().await;
        });
    });
    
    group.finish();
}

// Helper functions
fn generate_test_documents(count: usize, size: usize) -> Vec<ParsedDocument> {
    let words = [
        "arbitration", "dispute", "resolution", "agreement", "contract",
        "terms", "conditions", "legal", "binding", "court", "jurisdiction",
        "mediation", "settlement", "damages", "liability", "warranty",
        "breach", "performance", "default", "clause", "provision", "section"
    ];
    
    (0..count)
        .map(|i| {
            let mut content = String::new();
            let mut rng = (i as u64).wrapping_mul(1103515245).wrapping_add(12345);
            
            while content.len() < size {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let word_idx = (rng % words.len() as u64) as usize;
                content.push_str(words[word_idx]);
                content.push(' ');
                
                if content.len() % 100 == 0 {
                    content.push_str(".\n");
                }
            }
            content.truncate(size);
            
            create_test_document(i, content)
        })
        .collect()
}

fn generate_repetitive_documents(count: usize, size: usize) -> Vec<ParsedDocument> {
    let repetitive_content = "This is a highly repetitive document content that should compress very well. \
                            The arbitration clause in this agreement states that all disputes must be resolved \
                            through binding arbitration. This content repeats many times to test compression. ";
    
    (0..count)
        .map(|i| {
            let content = repetitive_content.repeat(size / repetitive_content.len() + 1);
            let content = content[..size].to_string();
            create_test_document(i, content)
        })
        .collect()
}

fn generate_documents_with_typos(count: usize, size: usize) -> Vec<ParsedDocument> {
    let words_with_typos = [
        "arbitration", "arbitraton", "arbitrtion", "arbtration",
        "dispute", "dispuite", "dispte", "disspute",
        "resolution", "resoltion", "resoluton", "reolution",
        "agreement", "agrement", "agreemnt", "agremeent",
        "contract", "contracct", "contrct", "contractt"
    ];
    
    (0..count)
        .map(|i| {
            let mut content = String::new();
            let mut rng = (i as u64 * 7919).wrapping_add(12345);
            
            while content.len() < size {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let word_idx = (rng % words_with_typos.len() as u64) as usize;
                content.push_str(words_with_typos[word_idx]);
                content.push(' ');
                
                if content.len() % 80 == 0 {
                    content.push_str(".\n");
                }
            }
            content.truncate(size);
            
            create_test_document(i, content)
        })
        .collect()
}

fn create_test_document(id: usize, content: String) -> ParsedDocument {
    use rust_core::{DocumentMetadata, DocumentStructure, ParseStatistics, FileType};
    
    ParsedDocument {
        id: format!("doc_{}", id),
        content: content.clone(),
        metadata: DocumentMetadata {
            file_type: FileType::PlainText,
            size: content.len(),
            encoding: "UTF-8".to_string(),
            language: Some("en".to_string()),
            created_at: Some(chrono::Utc::now()),
        },
        structure: DocumentStructure {
            paragraphs: Vec::new(),
            sections: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
        },
        statistics: ParseStatistics {
            total_chars: content.len(),
            total_words: content.split_whitespace().count(),
            total_lines: content.lines().count(),
            parse_time_ms: 0,
            memory_used: content.len(),
        },
    }
}

criterion_group!(
    benches,
    bench_document_indexing,
    bench_bulk_indexing,
    bench_search_performance,
    bench_memory_mapping,
    bench_compression,
    bench_fuzzy_search,
    bench_index_optimization
);
criterion_main!(benches);