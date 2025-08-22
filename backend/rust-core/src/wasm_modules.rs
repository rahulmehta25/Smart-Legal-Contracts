//! WebAssembly modules for client-side processing

use crate::{CoreError, CoreResult, DocumentParser, PatternMatcher, ParsedDocument, Pattern, PatternMatch};
use js_sys::{Array, Object, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{console, window, Worker, WorkerGlobalScope, MessageEvent};

/// WASM module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    pub enable_threading: bool,
    pub max_workers: usize,
    pub chunk_size: usize,
    pub enable_caching: bool,
    pub log_level: String,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            enable_threading: true,
            max_workers: 4,
            chunk_size: 64 * 1024,
            enable_caching: true,
            log_level: "info".to_string(),
        }
    }
}

/// Document processing result for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmProcessingResult {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
    pub processing_time_ms: u32,
    pub memory_used: usize,
}

/// Pattern matching result for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPatternResult {
    pub matches: Vec<WasmPatternMatch>,
    pub total_matches: usize,
    pub processing_time_ms: u32,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPatternMatch {
    pub pattern: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
    pub match_type: String,
}

/// WASM-compatible document processor
#[wasm_bindgen]
pub struct WasmDocumentProcessor {
    config: WasmConfig,
    parser: DocumentParser,
    pattern_matcher: PatternMatcher,
    cache: HashMap<String, Vec<u8>>,
}

/// WASM-compatible pattern matcher
#[wasm_bindgen]
pub struct WasmPatternMatcher {
    matcher: PatternMatcher,
    patterns: Vec<Pattern>,
}

/// WASM-compatible ML inference engine
#[wasm_bindgen]
pub struct WasmMLEngine {
    models: HashMap<String, Vec<u8>>,
    config: WasmConfig,
}

/// Worker pool for parallel processing
#[wasm_bindgen]
pub struct WasmWorkerPool {
    workers: Vec<Worker>,
    available_workers: Vec<usize>,
    config: WasmConfig,
}

#[wasm_bindgen]
impl WasmDocumentProcessor {
    /// Create new WASM document processor
    #[wasm_bindgen(constructor)]
    pub fn new(config_js: JsValue) -> Result<WasmDocumentProcessor, JsValue> {
        let config: WasmConfig = if config_js.is_undefined() || config_js.is_null() {
            WasmConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config_js)
                .map_err(|e| JsValue::from_str(&format!("Config parsing error: {}", e)))?
        };

        let parser_config = crate::parser::ParserConfig {
            enable_parallel: config.enable_threading,
            chunk_size: config.chunk_size,
            ..Default::default()
        };

        let pattern_config = crate::pattern_matcher::PatternConfig {
            parallel_threshold: config.chunk_size,
            ..Default::default()
        };

        let parser = DocumentParser::new(parser_config);
        let pattern_matcher = PatternMatcher::new(pattern_config);

        Ok(WasmDocumentProcessor {
            config,
            parser,
            pattern_matcher,
            cache: HashMap::new(),
        })
    }

    /// Parse document from text
    #[wasm_bindgen]
    pub fn parse_text(&mut self, text: &str, file_type: Option<String>) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        // Create a mock file for parsing
        let temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| JsValue::from_str(&format!("Temp file creation failed: {}", e)))?;
        
        std::fs::write(temp_file.path(), text)
            .map_err(|e| JsValue::from_str(&format!("File write failed: {}", e)))?;

        let result = self.parser.parse_file(temp_file.path())
            .map_err(|e| JsValue::from_str(&format!("Parsing failed: {}", e)))?;

        let processing_time = (web_sys::window().unwrap().performance().unwrap().now() - start_time) as u32;
        let memory_used = text.len();

        let wasm_result = WasmProcessingResult {
            success: true,
            data: Some(serde_json::to_value(&result).unwrap()),
            error: None,
            processing_time_ms: processing_time,
            memory_used,
        };

        serde_wasm_bindgen::to_value(&wasm_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Parse document from Uint8Array
    #[wasm_bindgen]
    pub fn parse_bytes(&mut self, bytes: Uint8Array) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let data: Vec<u8> = bytes.to_vec();
        let text = String::from_utf8(data)
            .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion failed: {}", e)))?;

        self.parse_text(&text, None)
    }

    /// Process document in chunks for large files
    #[wasm_bindgen]
    pub async fn process_large_document(&mut self, text: &str) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let chunks: Vec<&str> = text
            .as_bytes()
            .chunks(self.config.chunk_size)
            .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
            .collect();

        let mut results = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_result = self.parse_text(chunk, None)?;
            results.push(chunk_result);
            
            // Yield control to allow other operations
            if i % 10 == 0 {
                let promise = js_sys::Promise::resolve(&JsValue::from(0));
                wasm_bindgen_futures::JsFuture::from(promise).await.ok();
            }
        }

        let processing_time = (web_sys::window().unwrap().performance().unwrap().now() - start_time) as u32;

        let combined_result = WasmProcessingResult {
            success: true,
            data: Some(serde_json::json!({
                "chunks": results.len(),
                "results": results
            })),
            error: None,
            processing_time_ms: processing_time,
            memory_used: text.len(),
        };

        serde_wasm_bindgen::to_value(&combined_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Extract text from various formats
    #[wasm_bindgen]
    pub fn extract_text(&self, data: Uint8Array, mime_type: &str) -> Result<String, JsValue> {
        let bytes: Vec<u8> = data.to_vec();
        
        match mime_type {
            "text/plain" => {
                String::from_utf8(bytes)
                    .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion failed: {}", e)))
            }
            "application/json" => {
                let json_str = String::from_utf8(bytes)
                    .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion failed: {}", e)))?;
                
                // Extract text content from JSON
                let json: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| JsValue::from_str(&format!("JSON parsing failed: {}", e)))?;
                
                Ok(self.extract_text_from_json(&json))
            }
            "text/html" => {
                let html = String::from_utf8(bytes)
                    .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion failed: {}", e)))?;
                
                Ok(self.extract_text_from_html(&html))
            }
            _ => {
                Err(JsValue::from_str(&format!("Unsupported MIME type: {}", mime_type)))
            }
        }
    }

    /// Get processing statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let stats = serde_json::json!({
            "cache_size": self.cache.len(),
            "config": self.config,
            "memory_usage": self.estimate_memory_usage(),
        });

        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Clear processing cache
    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // Private helper methods
    fn extract_text_from_json(&self, json: &serde_json::Value) -> String {
        match json {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(obj) => {
                obj.values()
                    .map(|v| self.extract_text_from_json(v))
                    .collect::<Vec<String>>()
                    .join(" ")
            }
            serde_json::Value::Array(arr) => {
                arr.iter()
                    .map(|v| self.extract_text_from_json(v))
                    .collect::<Vec<String>>()
                    .join(" ")
            }
            _ => String::new(),
        }
    }

    fn extract_text_from_html(&self, html: &str) -> String {
        // Simple HTML tag removal (in production, use a proper HTML parser)
        let re = regex::Regex::new(r"<[^>]*>").unwrap();
        re.replace_all(html, " ").to_string()
    }

    fn estimate_memory_usage(&self) -> usize {
        self.cache.values().map(|v| v.len()).sum::<usize>()
            + std::mem::size_of::<Self>()
    }
}

#[wasm_bindgen]
impl WasmPatternMatcher {
    /// Create new WASM pattern matcher
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmPatternMatcher {
        let config = crate::pattern_matcher::PatternConfig::default();
        let matcher = PatternMatcher::new(config);
        
        WasmPatternMatcher {
            matcher,
            patterns: Vec::new(),
        }
    }

    /// Add literal pattern
    #[wasm_bindgen]
    pub fn add_literal_pattern(&mut self, pattern: &str) -> Result<(), JsValue> {
        self.patterns.push(Pattern::Literal(pattern.to_string()));
        Ok(())
    }

    /// Add regex pattern
    #[wasm_bindgen]
    pub fn add_regex_pattern(&mut self, pattern: &str) -> Result<(), JsValue> {
        self.patterns.push(Pattern::Regex(pattern.to_string()));
        Ok(())
    }

    /// Add fuzzy pattern
    #[wasm_bindgen]
    pub fn add_fuzzy_pattern(&mut self, pattern: &str, max_distance: usize) -> Result<(), JsValue> {
        self.patterns.push(Pattern::Fuzzy {
            text: pattern.to_string(),
            max_distance,
        });
        Ok(())
    }

    /// Compile patterns for matching
    #[wasm_bindgen]
    pub fn compile_patterns(&mut self) -> Result<(), JsValue> {
        self.matcher.compile_patterns(self.patterns.clone())
            .map_err(|e| JsValue::from_str(&format!("Pattern compilation failed: {}", e)))?;
        Ok(())
    }

    /// Find matches in text
    #[wasm_bindgen]
    pub fn find_matches(&self, text: &str) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let matches = self.matcher.find_matches(text)
            .map_err(|e| JsValue::from_str(&format!("Pattern matching failed: {}", e)))?;

        let processing_time = (web_sys::window().unwrap().performance().unwrap().now() - start_time) as u32;
        
        let wasm_matches: Vec<WasmPatternMatch> = matches
            .iter()
            .map(|m| WasmPatternMatch {
                pattern: m.pattern.clone(),
                text: m.matched_text.clone(),
                start: m.start,
                end: m.end,
                confidence: m.confidence,
                match_type: format!("{:?}", m.match_type),
            })
            .collect();

        let confidence_score = if !wasm_matches.is_empty() {
            wasm_matches.iter().map(|m| m.confidence).sum::<f64>() / wasm_matches.len() as f64
        } else {
            0.0
        };

        let result = WasmPatternResult {
            matches: wasm_matches.clone(),
            total_matches: wasm_matches.len(),
            processing_time_ms: processing_time,
            confidence_score,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Find matches with highlighting
    #[wasm_bindgen]
    pub fn find_matches_with_highlighting(&self, text: &str, highlight_class: &str) -> Result<JsValue, JsValue> {
        let matches_result = self.find_matches(text)?;
        let matches: WasmPatternResult = serde_wasm_bindgen::from_value(matches_result)?;
        
        let mut highlighted_text = text.to_string();
        let mut offset = 0i32;

        // Sort matches by start position (reverse order for correct offset calculation)
        let mut sorted_matches = matches.matches.clone();
        sorted_matches.sort_by(|a, b| b.start.cmp(&a.start));

        for m in &sorted_matches {
            let start = (m.start as i32 + offset) as usize;
            let end = (m.end as i32 + offset) as usize;
            
            if start <= highlighted_text.len() && end <= highlighted_text.len() {
                let before = &highlighted_text[..start];
                let matched = &highlighted_text[start..end];
                let after = &highlighted_text[end..];
                
                let highlighted = format!(
                    r#"{}<span class="{}" data-confidence="{:.2}">{}</span>{}"#,
                    before, highlight_class, m.confidence, matched, after
                );
                
                offset += highlighted.len() as i32 - highlighted_text.len() as i32;
                highlighted_text = highlighted;
            }
        }

        let result = serde_json::json!({
            "highlighted_text": highlighted_text,
            "matches": matches.matches,
            "total_matches": matches.total_matches,
            "processing_time_ms": matches.processing_time_ms,
            "confidence_score": matches.confidence_score
        });

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Get pattern statistics
    #[wasm_bindgen]
    pub fn get_pattern_stats(&self) -> Result<JsValue, JsValue> {
        let stats = serde_json::json!({
            "total_patterns": self.patterns.len(),
            "pattern_types": self.get_pattern_type_counts(),
        });

        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    fn get_pattern_type_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for pattern in &self.patterns {
            let pattern_type = match pattern {
                Pattern::Literal(_) => "literal",
                Pattern::Regex(_) => "regex",
                Pattern::Fuzzy { .. } => "fuzzy",
                Pattern::Compound { .. } => "compound",
            };
            *counts.entry(pattern_type.to_string()).or_insert(0) += 1;
        }
        counts
    }
}

#[wasm_bindgen]
impl WasmMLEngine {
    /// Create new WASM ML engine
    #[wasm_bindgen(constructor)]
    pub fn new(config_js: JsValue) -> Result<WasmMLEngine, JsValue> {
        let config: WasmConfig = if config_js.is_undefined() || config_js.is_null() {
            WasmConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config_js)
                .map_err(|e| JsValue::from_str(&format!("Config parsing error: {}", e)))?
        };

        Ok(WasmMLEngine {
            models: HashMap::new(),
            config,
        })
    }

    /// Load ML model from bytes
    #[wasm_bindgen]
    pub fn load_model(&mut self, model_id: &str, model_data: Uint8Array) -> Result<(), JsValue> {
        let bytes: Vec<u8> = model_data.to_vec();
        self.models.insert(model_id.to_string(), bytes);
        
        console::log_1(&format!("Loaded model: {} ({} bytes)", model_id, model_data.length()).into());
        Ok(())
    }

    /// Run inference on text
    #[wasm_bindgen]
    pub fn predict(&self, model_id: &str, text: &str) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        if !self.models.contains_key(model_id) {
            return Err(JsValue::from_str(&format!("Model not found: {}", model_id)));
        }

        // Simplified inference (in production, would use ONNX.js or TensorFlow.js)
        let mock_predictions = self.mock_inference(text);
        
        let processing_time = (web_sys::window().unwrap().performance().unwrap().now() - start_time) as u32;
        
        let result = serde_json::json!({
            "predictions": mock_predictions,
            "model_id": model_id,
            "processing_time_ms": processing_time,
            "input_length": text.len()
        });

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Batch prediction for multiple texts
    #[wasm_bindgen]
    pub async fn batch_predict(&self, model_id: &str, texts: JsValue) -> Result<JsValue, JsValue> {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let texts_array: Vec<String> = serde_wasm_bindgen::from_value(texts)
            .map_err(|e| JsValue::from_str(&format!("Text array parsing failed: {}", e)))?;

        let mut results = Vec::new();
        
        for (i, text) in texts_array.iter().enumerate() {
            let prediction = self.predict(model_id, text)?;
            results.push(prediction);
            
            // Yield control periodically
            if i % 10 == 0 {
                let promise = js_sys::Promise::resolve(&JsValue::from(0));
                wasm_bindgen_futures::JsFuture::from(promise).await.ok();
            }
        }

        let processing_time = (web_sys::window().unwrap().performance().unwrap().now() - start_time) as u32;
        
        let batch_result = serde_json::json!({
            "results": results,
            "batch_size": texts_array.len(),
            "processing_time_ms": processing_time
        });

        serde_wasm_bindgen::to_value(&batch_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Get available models
    #[wasm_bindgen]
    pub fn list_models(&self) -> Result<JsValue, JsValue> {
        let model_info: HashMap<String, serde_json::Value> = self.models
            .iter()
            .map(|(id, data)| {
                (id.clone(), serde_json::json!({
                    "size_bytes": data.len(),
                    "loaded": true
                }))
            })
            .collect();

        serde_wasm_bindgen::to_value(&model_info)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Remove model from memory
    #[wasm_bindgen]
    pub fn unload_model(&mut self, model_id: &str) -> bool {
        self.models.remove(model_id).is_some()
    }

    // Mock inference for demonstration
    fn mock_inference(&self, text: &str) -> Vec<f64> {
        // Simple mock: return character frequency as "features"
        let mut char_counts = vec![0.0; 26];
        let total_chars = text.len() as f64;
        
        for c in text.to_lowercase().chars() {
            if c.is_ascii_lowercase() {
                let idx = (c as u8 - b'a') as usize;
                if idx < 26 {
                    char_counts[idx] += 1.0;
                }
            }
        }
        
        // Normalize to frequencies
        if total_chars > 0.0 {
            for count in &mut char_counts {
                *count /= total_chars;
            }
        }
        
        char_counts
    }
}

/// JavaScript utility functions
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    
    console::log_1(&"Rust WASM module initialized".into());
}

/// Utility function to get memory usage
#[wasm_bindgen]
pub fn get_memory_usage() -> usize {
    use wasm_bindgen::memory;
    
    let mem = memory();
    mem.buffer().byte_length() as usize
}

/// Utility function to trigger garbage collection
#[wasm_bindgen]
pub fn trigger_gc() {
    // Note: WebAssembly doesn't have direct GC control
    // This is more of a placeholder for potential future use
    console::log_1(&"GC trigger requested".into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_wasm_pattern_matcher() {
        let mut matcher = WasmPatternMatcher::new();
        matcher.add_literal_pattern("test").unwrap();
        matcher.compile_patterns().unwrap();
        
        let result = matcher.find_matches("This is a test document").unwrap();
        // Would need to test the actual result structure
        assert!(!result.is_undefined());
    }

    #[wasm_bindgen_test]
    fn test_wasm_ml_engine() {
        let engine = WasmMLEngine::new(JsValue::NULL).unwrap();
        let models = engine.list_models().unwrap();
        assert!(!models.is_undefined());
    }
}