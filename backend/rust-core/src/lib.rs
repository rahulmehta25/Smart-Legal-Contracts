//! High-performance Rust core for critical operations
//! 
//! This crate provides optimized implementations for document parsing,
//! pattern matching, indexing, caching, and cryptographic operations.

pub mod parser;
pub mod pattern_matcher;
pub mod indexer;
pub mod cache;
pub mod crypto;
pub mod wasm_modules;
pub mod concurrency;
pub mod python_bindings;

pub use parser::*;
pub use pattern_matcher::*;
pub use indexer::*;
pub use cache::*;
pub use crypto::*;

use std::error::Error;
use std::fmt;

/// Core result type for all operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Core error types
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Pattern matching error: {0}")]
    PatternError(String),
    #[error("Indexing error: {0}")]
    IndexError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Crypto error: {0}")]
    CryptoError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Core configuration for all modules
#[derive(Debug, Clone)]
pub struct CoreConfig {
    pub max_workers: usize,
    pub cache_size: usize,
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub memory_limit: usize,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get(),
            cache_size: 1024 * 1024 * 100, // 100MB
            enable_simd: true,
            enable_parallel: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Initialize the core library with configuration
pub fn init(config: CoreConfig) -> CoreResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter("rust_core=debug")
        .init();
    
    tracing::info!("Initializing Rust Core with config: {:?}", config);
    
    // Initialize global state if needed
    cache::init_global_cache(config.cache_size)?;
    
    Ok(())
}