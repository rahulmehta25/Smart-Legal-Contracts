//! Parsing performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_core::{DocumentParser, ParserConfig};
use std::fs;
use tempfile::NamedTempFile;

/// Benchmark document parsing performance
fn bench_document_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_parsing");
    
    // Test different document sizes
    let sizes = [1024, 10_240, 102_400, 1_024_000]; // 1KB, 10KB, 100KB, 1MB
    
    for size in sizes.iter() {
        let content = generate_test_content(*size);
        let temp_file = create_temp_file(&content);
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        // Benchmark sequential parsing
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            size,
            |b, _| {
                let config = ParserConfig {
                    enable_parallel: false,
                    memory_mapped: false,
                    ..Default::default()
                };
                let parser = DocumentParser::new(config);
                
                b.iter(|| {
                    let _ = parser.parse_file(black_box(temp_file.path()));
                });
            },
        );
        
        // Benchmark parallel parsing
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            size,
            |b, _| {
                let config = ParserConfig {
                    enable_parallel: true,
                    memory_mapped: false,
                    ..Default::default()
                };
                let parser = DocumentParser::new(config);
                
                b.iter(|| {
                    let _ = parser.parse_file(black_box(temp_file.path()));
                });
            },
        );
        
        // Benchmark memory-mapped parsing
        group.bench_with_input(
            BenchmarkId::new("memory_mapped", size),
            size,
            |b, _| {
                let config = ParserConfig {
                    enable_parallel: true,
                    memory_mapped: true,
                    ..Default::default()
                };
                let parser = DocumentParser::new(config);
                
                b.iter(|| {
                    let _ = parser.parse_file(black_box(temp_file.path()));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD vs non-SIMD parsing
fn bench_simd_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_parsing");
    
    let content = generate_test_content(100_000); // 100KB
    let temp_file = create_temp_file(&content);
    
    group.throughput(Throughput::Bytes(content.len() as u64));
    
    // Regular parsing
    group.bench_function("regular", |b| {
        let config = ParserConfig {
            chunk_size: 1024,
            enable_parallel: false,
            memory_mapped: false,
            ..Default::default()
        };
        let parser = DocumentParser::new(config);
        
        b.iter(|| {
            let _ = parser.parse_file(black_box(temp_file.path()));
        });
    });
    
    // SIMD-optimized parsing (simulated - would need actual SIMD implementation)
    group.bench_function("simd_optimized", |b| {
        let config = ParserConfig {
            chunk_size: 32, // SIMD width
            enable_parallel: false,
            memory_mapped: true,
            ..Default::default()
        };
        let parser = DocumentParser::new(config);
        
        b.iter(|| {
            let _ = parser.parse_file(black_box(temp_file.path()));
        });
    });
    
    group.finish();
}

/// Benchmark parsing different file types
fn bench_file_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_types");
    
    let file_types = [
        ("plain_text", generate_plain_text()),
        ("json", generate_json_content()),
        ("html", generate_html_content()),
        ("markdown", generate_markdown_content()),
    ];
    
    for (name, content) in file_types.iter() {
        let temp_file = create_temp_file(content);
        
        group.throughput(Throughput::Bytes(content.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("parse_file_type", name),
            name,
            |b, _| {
                let config = ParserConfig::default();
                let parser = DocumentParser::new(config);
                
                b.iter(|| {
                    let _ = parser.parse_file(black_box(temp_file.path()));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark streaming vs batch parsing
fn bench_streaming_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_parsing");
    
    let content = generate_test_content(500_000); // 500KB
    group.throughput(Throughput::Bytes(content.len() as u64));
    
    // Batch parsing
    group.bench_function("batch", |b| {
        let temp_file = create_temp_file(&content);
        let config = ParserConfig::default();
        let parser = DocumentParser::new(config);
        
        b.iter(|| {
            let _ = parser.parse_file(black_box(temp_file.path()));
        });
    });
    
    // Streaming parsing
    group.bench_function("streaming", |b| {
        let config = ParserConfig {
            enable_streaming: true,
            chunk_size: 8192,
            ..Default::default()
        };
        let parser = DocumentParser::new(config);
        
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            let cursor = std::io::Cursor::new(content.as_bytes());
            let _ = parser.parse_stream(black_box(cursor)).await;
        });
    });
    
    group.finish();
}

// Helper functions
fn generate_test_content(size: usize) -> String {
    let paragraph = "This is a sample paragraph with various words that should be parsed and analyzed. \
                    It contains different types of content including punctuation, numbers like 123 and 456, \
                    and various formatting elements. The document parser should handle all of this content \
                    efficiently and extract meaningful structure from the text.\n\n";
    
    let paragraphs_needed = (size / paragraph.len()) + 1;
    paragraph.repeat(paragraphs_needed)[..size].to_string()
}

fn generate_plain_text() -> String {
    "This is a plain text document.\nIt has multiple lines.\nWith different content.\n".repeat(1000)
}

fn generate_json_content() -> String {
    serde_json::json!({
        "title": "Test Document",
        "content": "This is the main content of the document",
        "metadata": {
            "author": "Test Author",
            "created_at": "2024-01-01T00:00:00Z",
            "tags": ["test", "benchmark", "json"]
        },
        "sections": [
            {
                "title": "Introduction",
                "content": "This is the introduction section"
            },
            {
                "title": "Main Content", 
                "content": "This is the main content section with a lot of text that should be parsed"
            }
        ]
    }).to_string().repeat(100)
}

fn generate_html_content() -> String {
    r#"<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
        <li>List item 3</li>
    </ul>
    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
        </tr>
    </table>
</body>
</html>"#.repeat(100)
}

fn generate_markdown_content() -> String {
    r#"# Main Title

This is a paragraph with **bold text** and *italic text*.

## Section 1

Here's some content with a [link](http://example.com).

### Subsection

- List item 1
- List item 2
- List item 3

```rust
fn hello_world() {
    println!("Hello, world!");
}
```

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

"#.repeat(100)
}

fn create_temp_file(content: &str) -> NamedTempFile {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    fs::write(temp_file.path(), content).expect("Failed to write temp file");
    temp_file
}

criterion_group!(
    benches,
    bench_document_parsing,
    bench_simd_parsing,
    bench_file_types,
    bench_streaming_parsing
);
criterion_main!(benches);