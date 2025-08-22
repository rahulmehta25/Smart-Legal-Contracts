//! High-speed pattern matching engine with multiple algorithms

use crate::{CoreError, CoreResult};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use edit_distance::edit_distance;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use suffix::SuffixTable;

/// Pattern matching configuration
#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub case_sensitive: bool,
    pub whole_words: bool,
    pub max_edit_distance: usize,
    pub enable_fuzzy: bool,
    pub enable_regex: bool,
    pub parallel_threshold: usize,
    pub max_patterns: usize,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            whole_words: false,
            max_edit_distance: 2,
            enable_fuzzy: true,
            enable_regex: true,
            parallel_threshold: 1000,
            max_patterns: 10000,
        }
    }
}

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: String,
    pub matched_text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
    pub match_type: MatchType,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Fuzzy { distance: usize },
    Regex,
    Substring,
    WholeWord,
}

/// Pattern types for different matching strategies
#[derive(Debug, Clone)]
pub enum Pattern {
    Literal(String),
    Regex(String),
    Fuzzy { text: String, max_distance: usize },
    Compound { patterns: Vec<Pattern>, operator: LogicalOperator },
}

#[derive(Debug, Clone)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// High-performance pattern matcher
pub struct PatternMatcher {
    config: PatternConfig,
    aho_corasick: Option<AhoCorasick>,
    regex_engine: Option<RegexEngine>,
    suffix_array: Option<SuffixArray>,
    compiled_patterns: Vec<CompiledPattern>,
}

#[derive(Debug, Clone)]
struct CompiledPattern {
    pattern: Pattern,
    id: usize,
    weight: f64,
}

/// Specialized regex engine with optimizations
struct RegexEngine {
    compiled_regexes: HashMap<String, Regex>,
    cache_size: usize,
}

/// Suffix array for fast substring matching
struct SuffixArray {
    text: String,
    suffix_table: SuffixTable,
}

impl PatternMatcher {
    /// Create new pattern matcher with configuration
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            aho_corasick: None,
            regex_engine: Some(RegexEngine::new(1000)),
            suffix_array: None,
            compiled_patterns: Vec::new(),
        }
    }

    /// Compile patterns for optimized matching
    pub fn compile_patterns(&mut self, patterns: Vec<Pattern>) -> CoreResult<()> {
        if patterns.len() > self.config.max_patterns {
            return Err(CoreError::PatternError(
                format!("Too many patterns: {}", patterns.len())
            ));
        }

        // Separate literal patterns for Aho-Corasick
        let mut literal_patterns = Vec::new();
        let mut compiled_patterns = Vec::new();

        for (id, pattern) in patterns.into_iter().enumerate() {
            match &pattern {
                Pattern::Literal(text) => {
                    literal_patterns.push(if self.config.case_sensitive {
                        text.clone()
                    } else {
                        text.to_lowercase()
                    });
                }
                Pattern::Regex(regex_str) => {
                    if let Some(ref mut engine) = self.regex_engine {
                        engine.compile_regex(regex_str.clone())?;
                    }
                }
                _ => {}
            }

            compiled_patterns.push(CompiledPattern {
                pattern,
                id,
                weight: 1.0,
            });
        }

        // Build Aho-Corasick automaton for literal patterns
        if !literal_patterns.is_empty() {
            let mut builder = AhoCorasickBuilder::new();
            builder.match_kind(MatchKind::LeftmostLongest);
            
            if !self.config.case_sensitive {
                builder.ascii_case_insensitive(true);
            }

            self.aho_corasick = Some(
                builder
                    .build(literal_patterns)
                    .map_err(|e| CoreError::PatternError(e.to_string()))?
            );
        }

        self.compiled_patterns = compiled_patterns;
        Ok(())
    }

    /// Find all pattern matches in text
    pub fn find_matches(&self, text: &str) -> CoreResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let search_text = if self.config.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        // Aho-Corasick for literal patterns
        if let Some(ref ac) = self.aho_corasick {
            let ac_matches = self.find_aho_corasick_matches(&search_text, ac)?;
            matches.extend(ac_matches);
        }

        // Regex patterns
        if self.config.enable_regex {
            if let Some(ref engine) = self.regex_engine {
                let regex_matches = engine.find_matches(&search_text)?;
                matches.extend(regex_matches);
            }
        }

        // Fuzzy matching
        if self.config.enable_fuzzy {
            let fuzzy_matches = self.find_fuzzy_matches(&search_text)?;
            matches.extend(fuzzy_matches);
        }

        // Parallel processing for large result sets
        if matches.len() > self.config.parallel_threshold {
            matches.par_sort_by(|a, b| {
                a.start.cmp(&b.start).then_with(|| {
                    b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
                })
            });
        } else {
            matches.sort_by(|a, b| {
                a.start.cmp(&b.start).then_with(|| {
                    b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
                })
            });
        }

        // Remove overlapping matches with lower confidence
        let filtered_matches = self.filter_overlapping_matches(matches);

        Ok(filtered_matches)
    }

    /// Parallel pattern matching for large texts
    pub fn find_matches_parallel(&self, text: &str, chunk_size: usize) -> CoreResult<Vec<PatternMatch>> {
        let chunks: Vec<_> = text
            .char_indices()
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|chunk| {
                let start_idx = chunk[0].0;
                let end_idx = chunk.last().map(|(idx, c)| idx + c.len_utf8()).unwrap_or(start_idx);
                (start_idx, &text[start_idx..end_idx])
            })
            .collect();

        let chunk_matches: Vec<Vec<PatternMatch>> = chunks
            .par_iter()
            .map(|(offset, chunk)| {
                let mut matches = self.find_matches(chunk).unwrap_or_default();
                // Adjust positions relative to original text
                for match_result in &mut matches {
                    match_result.start += offset;
                    match_result.end += offset;
                }
                matches
            })
            .collect();

        let mut all_matches = Vec::new();
        for matches in chunk_matches {
            all_matches.extend(matches);
        }

        // Merge overlapping matches between chunks
        all_matches.sort_by_key(|m| m.start);
        let filtered_matches = self.filter_overlapping_matches(all_matches);

        Ok(filtered_matches)
    }

    /// Boyer-Moore string search optimization
    pub fn boyer_moore_search(&self, text: &str, pattern: &str) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let pattern_bytes = pattern.as_bytes();
        let text_bytes = text.as_bytes();
        let mut matches = Vec::new();

        // Build bad character table
        let mut bad_char = vec![pattern_bytes.len(); 256];
        for (i, &byte) in pattern_bytes.iter().enumerate() {
            bad_char[byte as usize] = pattern_bytes.len() - 1 - i;
        }

        let mut skip = 0;
        while skip <= text_bytes.len().saturating_sub(pattern_bytes.len()) {
            let mut j = pattern_bytes.len() - 1;
            
            while j < pattern_bytes.len() && pattern_bytes[j] == text_bytes[skip + j] {
                if j == 0 {
                    matches.push(skip);
                    break;
                }
                j -= 1;
            }

            if j < pattern_bytes.len() {
                skip += std::cmp::max(1, bad_char[text_bytes[skip + j] as usize]);
            } else {
                skip += 1;
            }
        }

        matches
    }

    /// Find matches using Aho-Corasick
    fn find_aho_corasick_matches(&self, text: &str, ac: &AhoCorasick) -> CoreResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();

        for mat in ac.find_iter(text) {
            let pattern_id = mat.pattern().as_usize();
            let pattern_text = &self.compiled_patterns[pattern_id];
            
            let matched_text = &text[mat.start()..mat.end()];
            let context = self.extract_context(text, mat.start(), mat.end());

            matches.push(PatternMatch {
                pattern: format!("{:?}", pattern_text.pattern),
                matched_text: matched_text.to_string(),
                start: mat.start(),
                end: mat.end(),
                confidence: 1.0,
                match_type: MatchType::Exact,
                context,
            });
        }

        Ok(matches)
    }

    /// Find fuzzy matches with edit distance
    fn find_fuzzy_matches(&self, text: &str) -> CoreResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_pos = 0;

        for pattern in &self.compiled_patterns {
            if let Pattern::Fuzzy { text: pattern_text, max_distance } = &pattern.pattern {
                for word in &words {
                    let distance = edit_distance(pattern_text, word);
                    
                    if distance <= *max_distance {
                        // Find the actual position of this word in the text
                        if let Some(word_start) = text[current_pos..].find(word) {
                            let absolute_start = current_pos + word_start;
                            let absolute_end = absolute_start + word.len();
                            let context = self.extract_context(text, absolute_start, absolute_end);
                            
                            let confidence = 1.0 - (distance as f64 / max_distance.max(&1) as f64);
                            
                            matches.push(PatternMatch {
                                pattern: pattern_text.clone(),
                                matched_text: word.to_string(),
                                start: absolute_start,
                                end: absolute_end,
                                confidence,
                                match_type: MatchType::Fuzzy { distance },
                                context,
                            });
                        }
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Extract context around a match
    fn extract_context(&self, text: &str, start: usize, end: usize) -> Option<String> {
        let context_size = 50;
        let context_start = start.saturating_sub(context_size);
        let context_end = std::cmp::min(end + context_size, text.len());
        
        if context_start < context_end {
            Some(text[context_start..context_end].to_string())
        } else {
            None
        }
    }

    /// Filter overlapping matches, keeping higher confidence ones
    fn filter_overlapping_matches(&self, mut matches: Vec<PatternMatch>) -> Vec<PatternMatch> {
        if matches.is_empty() {
            return matches;
        }

        matches.sort_by_key(|m| m.start);
        let mut filtered = Vec::new();
        let mut last_end = 0;

        for current_match in matches {
            if current_match.start >= last_end {
                last_end = current_match.end;
                filtered.push(current_match);
            } else {
                // Check if current match has higher confidence
                if let Some(last_match) = filtered.last_mut() {
                    if current_match.confidence > last_match.confidence {
                        *last_match = current_match;
                        last_end = last_match.end;
                    }
                }
            }
        }

        filtered
    }

    /// Build suffix array for the given text
    pub fn build_suffix_array(&mut self, text: String) -> CoreResult<()> {
        let suffix_table = SuffixTable::new(&text);
        self.suffix_array = Some(SuffixArray {
            text: text.clone(),
            suffix_table,
        });
        Ok(())
    }

    /// Find patterns using suffix array
    pub fn suffix_array_search(&self, pattern: &str) -> CoreResult<Vec<usize>> {
        if let Some(ref suffix_array) = self.suffix_array {
            let positions = suffix_array.suffix_table.positions(pattern);
            Ok(positions)
        } else {
            Err(CoreError::PatternError("Suffix array not built".to_string()))
        }
    }

    /// Advanced pattern analysis with statistics
    pub fn analyze_patterns(&self, text: &str) -> CoreResult<PatternAnalysis> {
        let matches = self.find_matches(text)?;
        
        let total_matches = matches.len();
        let unique_patterns = matches
            .iter()
            .map(|m| &m.pattern)
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        let avg_confidence = if !matches.is_empty() {
            matches.iter().map(|m| m.confidence).sum::<f64>() / matches.len() as f64
        } else {
            0.0
        };

        let pattern_frequency = matches
            .iter()
            .fold(HashMap::new(), |mut acc, m| {
                *acc.entry(m.pattern.clone()).or_insert(0) += 1;
                acc
            });

        Ok(PatternAnalysis {
            total_matches,
            unique_patterns,
            average_confidence: avg_confidence,
            pattern_frequency,
            coverage_percentage: (matches.len() as f64 / text.len() as f64) * 100.0,
        })
    }
}

/// Pattern analysis results
#[derive(Debug, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub total_matches: usize,
    pub unique_patterns: usize,
    pub average_confidence: f64,
    pub pattern_frequency: HashMap<String, usize>,
    pub coverage_percentage: f64,
}

impl RegexEngine {
    fn new(cache_size: usize) -> Self {
        Self {
            compiled_regexes: HashMap::new(),
            cache_size,
        }
    }

    fn compile_regex(&mut self, pattern: String) -> CoreResult<()> {
        if !self.compiled_regexes.contains_key(&pattern) {
            // Manage cache size
            if self.compiled_regexes.len() >= self.cache_size {
                // Remove oldest entries (simplified - could use LRU)
                self.compiled_regexes.clear();
            }

            let regex = Regex::new(&pattern)
                .map_err(|e| CoreError::PatternError(e.to_string()))?;
            
            self.compiled_regexes.insert(pattern, regex);
        }
        Ok(())
    }

    fn find_matches(&self, text: &str) -> CoreResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();

        for (pattern_str, regex) in &self.compiled_regexes {
            for mat in regex.find_iter(text) {
                matches.push(PatternMatch {
                    pattern: pattern_str.clone(),
                    matched_text: mat.as_str().to_string(),
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    match_type: MatchType::Regex,
                    context: None,
                });
            }
        }

        Ok(matches)
    }
}

/// Parallel pattern search using work stealing
pub struct ParallelPatternSearcher {
    patterns: Arc<Vec<Pattern>>,
    config: PatternConfig,
}

impl ParallelPatternSearcher {
    pub fn new(patterns: Vec<Pattern>, config: PatternConfig) -> Self {
        Self {
            patterns: Arc::new(patterns),
            config,
        }
    }

    /// Search patterns across multiple documents in parallel
    pub fn search_documents(&self, documents: &[String]) -> CoreResult<Vec<Vec<PatternMatch>>> {
        let results: Vec<Vec<PatternMatch>> = documents
            .par_iter()
            .map(|doc| {
                let mut matcher = PatternMatcher::new(self.config.clone());
                matcher.compile_patterns((*self.patterns).clone()).unwrap();
                matcher.find_matches(doc).unwrap_or_default()
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matcher_creation() {
        let config = PatternConfig::default();
        let _matcher = PatternMatcher::new(config);
        assert!(true);
    }

    #[test]
    fn test_literal_pattern_matching() {
        let mut matcher = PatternMatcher::new(PatternConfig::default());
        let patterns = vec![
            Pattern::Literal("test".to_string()),
            Pattern::Literal("example".to_string()),
        ];
        
        matcher.compile_patterns(patterns).unwrap();
        let matches = matcher.find_matches("This is a test example text").unwrap();
        
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].matched_text, "test");
        assert_eq!(matches[1].matched_text, "example");
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut matcher = PatternMatcher::new(PatternConfig::default());
        let patterns = vec![
            Pattern::Fuzzy {
                text: "hello".to_string(),
                max_distance: 1,
            },
        ];
        
        matcher.compile_patterns(patterns).unwrap();
        let matches = matcher.find_matches("helo world").unwrap();
        
        assert!(!matches.is_empty());
        if let MatchType::Fuzzy { distance } = &matches[0].match_type {
            assert_eq!(*distance, 1);
        } else {
            panic!("Expected fuzzy match");
        }
    }

    #[test]
    fn test_boyer_moore_search() {
        let matcher = PatternMatcher::new(PatternConfig::default());
        let positions = matcher.boyer_moore_search("hello world hello", "hello");
        
        assert_eq!(positions, vec![0, 12]);
    }

    #[test]
    fn test_pattern_analysis() {
        let mut matcher = PatternMatcher::new(PatternConfig::default());
        let patterns = vec![
            Pattern::Literal("the".to_string()),
            Pattern::Literal("and".to_string()),
        ];
        
        matcher.compile_patterns(patterns).unwrap();
        let analysis = matcher.analyze_patterns("the quick brown fox and the lazy dog").unwrap();
        
        assert!(analysis.total_matches > 0);
        assert!(analysis.average_confidence > 0.0);
    }
}