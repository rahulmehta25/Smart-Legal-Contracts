//! Pattern matching performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_core::{PatternMatcher, PatternConfig, Pattern};
use std::time::Duration;

/// Benchmark pattern matching performance
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    
    let text_sizes = [1024, 10_240, 102_400, 1_024_000]; // 1KB to 1MB
    let pattern_counts = [1, 10, 100, 1000];
    
    for &text_size in &text_sizes {
        let text = generate_test_text(text_size);
        
        for &pattern_count in &pattern_counts {
            let patterns = generate_literal_patterns(pattern_count);
            
            group.throughput(Throughput::Bytes(text_size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}patterns", pattern_count), text_size),
                &(text, patterns),
                |b, (text, patterns)| {
                    let mut matcher = PatternMatcher::new(PatternConfig::default());
                    matcher.compile_patterns(patterns.clone()).unwrap();
                    
                    b.iter(|| {
                        let _ = matcher.find_matches(black_box(text));
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark different pattern matching algorithms
fn bench_pattern_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_algorithms");
    
    let text = generate_test_text(100_000); // 100KB
    group.throughput(Throughput::Bytes(text.len() as u64));
    
    // Aho-Corasick (multiple literal patterns)
    group.bench_function("aho_corasick", |b| {
        let patterns = generate_literal_patterns(50);
        let mut matcher = PatternMatcher::new(PatternConfig::default());
        matcher.compile_patterns(patterns).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    // Regex patterns
    group.bench_function("regex", |b| {
        let patterns = generate_regex_patterns(10);
        let config = PatternConfig {
            enable_regex: true,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    // Fuzzy matching
    group.bench_function("fuzzy", |b| {
        let patterns = generate_fuzzy_patterns(20);
        let config = PatternConfig {
            enable_fuzzy: true,
            max_edit_distance: 2,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    // Boyer-Moore single pattern
    group.bench_function("boyer_moore", |b| {
        let matcher = PatternMatcher::new(PatternConfig::default());
        let pattern = "arbitration";
        
        b.iter(|| {
            let _ = matcher.boyer_moore_search(black_box(&text), black_box(pattern));
        });
    });
    
    group.finish();
}

/// Benchmark parallel vs sequential pattern matching
fn bench_parallel_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_matching");
    
    let text = generate_test_text(1_000_000); // 1MB
    let patterns = generate_literal_patterns(100);
    
    group.throughput(Throughput::Bytes(text.len() as u64));
    
    // Sequential matching
    group.bench_function("sequential", |b| {
        let config = PatternConfig {
            enable_parallel: false,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns.clone()).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    // Parallel matching
    group.bench_function("parallel", |b| {
        let config = PatternConfig {
            enable_parallel: true,
            parallel_threshold: 1000,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns.clone()).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches_parallel(black_box(&text), 10000);
        });
    });
    
    group.finish();
}

/// Benchmark pattern compilation performance
fn bench_pattern_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_compilation");
    
    let pattern_counts = [10, 100, 1000, 10000];
    
    for &count in &pattern_counts {
        group.bench_with_input(
            BenchmarkId::new("literal", count),
            &count,
            |b, &count| {
                let patterns = generate_literal_patterns(count);
                
                b.iter(|| {
                    let mut matcher = PatternMatcher::new(PatternConfig::default());
                    let _ = matcher.compile_patterns(black_box(patterns.clone()));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("regex", std::cmp::min(count, 100)),
            &std::cmp::min(count, 100), // Limit regex patterns due to compilation cost
            |b, &count| {
                let patterns = generate_regex_patterns(count);
                
                b.iter(|| {
                    let config = PatternConfig {
                        enable_regex: true,
                        ..Default::default()
                    };
                    let mut matcher = PatternMatcher::new(config);
                    let _ = matcher.compile_patterns(black_box(patterns.clone()));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark case sensitivity impact
fn bench_case_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("case_sensitivity");
    
    let text = generate_mixed_case_text(50_000);
    let patterns = generate_mixed_case_patterns(50);
    
    group.throughput(Throughput::Bytes(text.len() as u64));
    
    // Case sensitive
    group.bench_function("case_sensitive", |b| {
        let config = PatternConfig {
            case_sensitive: true,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns.clone()).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    // Case insensitive
    group.bench_function("case_insensitive", |b| {
        let config = PatternConfig {
            case_sensitive: false,
            ..Default::default()
        };
        let mut matcher = PatternMatcher::new(config);
        matcher.compile_patterns(patterns.clone()).unwrap();
        
        b.iter(|| {
            let _ = matcher.find_matches(black_box(&text));
        });
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    let text = generate_test_text(100_000);
    
    // Test different pattern counts for memory impact
    let pattern_counts = [100, 500, 1000, 5000];
    
    for &count in &pattern_counts {
        let patterns = generate_literal_patterns(count);
        
        group.bench_with_input(
            BenchmarkId::new("pattern_count", count),
            &count,
            |b, _| {
                b.iter_with_setup(
                    || {
                        let mut matcher = PatternMatcher::new(PatternConfig::default());
                        matcher.compile_patterns(patterns.clone()).unwrap();
                        matcher
                    },
                    |matcher| {
                        let _ = matcher.find_matches(black_box(&text));
                    },
                );
            },
        );
    }
    
    group.finish();
}

// Helper functions
fn generate_test_text(size: usize) -> String {
    let words = [
        "arbitration", "clause", "dispute", "resolution", "agreement", 
        "contract", "terms", "conditions", "legal", "binding",
        "court", "jurisdiction", "mediation", "settlement", "damages",
        "liability", "warranty", "breach", "performance", "default"
    ];
    
    let mut text = String::new();
    let mut rng = 12345u64; // Simple LCG for reproducible results
    
    while text.len() < size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let word_idx = (rng % words.len() as u64) as usize;
        text.push_str(words[word_idx]);
        text.push(' ');
        
        if text.len() % 100 == 0 {
            text.push_str(".\n");
        }
    }
    
    text.truncate(size);
    text
}

fn generate_literal_patterns(count: usize) -> Vec<Pattern> {
    let base_patterns = [
        "arbitration", "dispute", "resolution", "agreement", "contract",
        "terms", "conditions", "legal", "binding", "court",
        "jurisdiction", "mediation", "settlement", "damages", "liability",
        "warranty", "breach", "performance", "default", "clause"
    ];
    
    (0..count)
        .map(|i| {
            let pattern = base_patterns[i % base_patterns.len()];
            if i >= base_patterns.len() {
                Pattern::Literal(format!("{}{}", pattern, i / base_patterns.len()))
            } else {
                Pattern::Literal(pattern.to_string())
            }
        })
        .collect()
}

fn generate_regex_patterns(count: usize) -> Vec<Pattern> {
    let base_regexes = [
        r"\barbitration\b",
        r"\bdispute\s+resolution\b",
        r"\bagreement\s+\w+",
        r"\bcontract\w*",
        r"\blegal\s+\w+",
        r"\bcourt\s+of\s+\w+",
        r"\bmediation\s+process",
        r"\bsettlement\s+amount",
        r"\bdamages\s+\$\d+",
        r"\bliability\s+limited"
    ];
    
    (0..count)
        .map(|i| {
            let regex = base_regexes[i % base_regexes.len()];
            Pattern::Regex(regex.to_string())
        })
        .collect()
}

fn generate_fuzzy_patterns(count: usize) -> Vec<Pattern> {
    let base_words = [
        "arbitration", "dispute", "resolution", "agreement", "contract",
        "mediation", "settlement", "liability", "warranty", "jurisdiction"
    ];
    
    (0..count)
        .map(|i| {
            let word = base_words[i % base_words.len()];
            Pattern::Fuzzy {
                text: word.to_string(),
                max_distance: 2,
            }
        })
        .collect()
}

fn generate_mixed_case_text(size: usize) -> String {
    let words = [
        "Arbitration", "CLAUSE", "dispute", "Resolution", "AGREEMENT",
        "Contract", "TERMS", "conditions", "Legal", "BINDING"
    ];
    
    let mut text = String::new();
    let mut rng = 54321u64;
    
    while text.len() < size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let word_idx = (rng % words.len() as u64) as usize;
        text.push_str(words[word_idx]);
        text.push(' ');
    }
    
    text.truncate(size);
    text
}

fn generate_mixed_case_patterns(count: usize) -> Vec<Pattern> {
    let base_patterns = [
        "arbitration", "Arbitration", "ARBITRATION",
        "clause", "Clause", "CLAUSE",
        "dispute", "Dispute", "DISPUTE",
        "agreement", "Agreement", "AGREEMENT"
    ];
    
    (0..count)
        .map(|i| Pattern::Literal(base_patterns[i % base_patterns.len()].to_string()))
        .collect()
}

criterion_group!(
    benches,
    bench_pattern_matching,
    bench_pattern_algorithms,
    bench_parallel_matching,
    bench_pattern_compilation,
    bench_case_sensitivity,
    bench_memory_usage
);
criterion_main!(benches);