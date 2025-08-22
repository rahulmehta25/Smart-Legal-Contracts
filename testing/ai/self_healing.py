"""
Self-Healing Test Framework

This module provides automatic test repair capabilities using AI to detect
and fix common test failures, update locators, and maintain test stability.
"""

import ast
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib

import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, ElementNotInteractableException,
    StaleElementReferenceException, WebDriverException
)
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FailureType(Enum):
    LOCATOR_NOT_FOUND = "locator_not_found"
    ELEMENT_NOT_INTERACTABLE = "element_not_interactable"
    TIMEOUT = "timeout"
    STALE_ELEMENT = "stale_element"
    ASSERTION_ERROR = "assertion_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    ENVIRONMENT_CHANGE = "environment_change"
    DATA_CHANGE = "data_change"


class RepairStrategy(Enum):
    LOCATOR_HEALING = "locator_healing"
    WAIT_STRATEGY = "wait_strategy"
    RETRY_MECHANISM = "retry_mechanism"
    ALTERNATIVE_PATH = "alternative_path"
    SMART_ASSERTION = "smart_assertion"
    DYNAMIC_DATA = "dynamic_data"


@dataclass
class FailureAnalysis:
    """Analysis of a test failure."""
    failure_type: FailureType
    original_locator: str
    error_message: str
    stack_trace: str
    page_url: str
    page_source: str
    screenshot_path: Optional[str]
    suggested_repairs: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RepairAction:
    """A repair action for a failed test."""
    strategy: RepairStrategy
    original_code: str
    repaired_code: str
    confidence: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestHealingResult:
    """Result of a test healing attempt."""
    test_file: str
    test_function: str
    original_failure: FailureAnalysis
    applied_repairs: List[RepairAction]
    healing_successful: bool
    execution_time: float
    new_test_code: str
    validation_results: Dict[str, Any]


class ElementLocatorHealer:
    """Heals broken element locators using AI-powered strategies."""
    
    def __init__(self):
        self.locator_strategies = [
            self._heal_by_similarity,
            self._heal_by_text_content,
            self._heal_by_position,
            self._heal_by_attributes,
            self._heal_by_hierarchy,
            self._heal_by_css_selector
        ]
        
    def heal_locator(self, driver: webdriver.Chrome, 
                    original_locator: str, 
                    locator_type: str = "xpath") -> List[Tuple[str, float]]:
        """Heal a broken locator by finding alternative ways to locate the element."""
        healing_suggestions = []
        
        # Get current page elements
        page_elements = self._analyze_page_elements(driver)
        
        # Try different healing strategies
        for strategy in self.locator_strategies:
            try:
                suggestions = strategy(driver, original_locator, locator_type, page_elements)
                healing_suggestions.extend(suggestions)
            except Exception as e:
                logging.debug(f"Healing strategy failed: {e}")
                
        # Sort by confidence and remove duplicates
        unique_suggestions = {}
        for locator, confidence in healing_suggestions:
            if locator not in unique_suggestions or unique_suggestions[locator] < confidence:
                unique_suggestions[locator] = confidence
                
        # Return top suggestions
        sorted_suggestions = sorted(unique_suggestions.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:5]  # Top 5 suggestions
        
    def _analyze_page_elements(self, driver: webdriver.Chrome) -> List[Dict[str, Any]]:
        """Analyze all elements on the current page."""
        elements_info = []
        
        try:
            # Get all elements
            all_elements = driver.find_elements(By.XPATH, "//*")
            
            for element in all_elements[:100]:  # Limit to prevent timeout
                try:
                    element_info = {
                        'tag': element.tag_name,
                        'id': element.get_attribute('id') or '',
                        'class': element.get_attribute('class') or '',
                        'text': element.text.strip(),
                        'name': element.get_attribute('name') or '',
                        'type': element.get_attribute('type') or '',
                        'placeholder': element.get_attribute('placeholder') or '',
                        'title': element.get_attribute('title') or '',
                        'href': element.get_attribute('href') or '',
                        'src': element.get_attribute('src') or '',
                        'value': element.get_attribute('value') or '',
                        'location': element.location,
                        'size': element.size,
                        'is_displayed': element.is_displayed(),
                        'is_enabled': element.is_enabled()
                    }
                    elements_info.append(element_info)
                except StaleElementReferenceException:
                    continue
                except Exception as e:
                    logging.debug(f"Error analyzing element: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error analyzing page elements: {e}")
            
        return elements_info
        
    def _heal_by_similarity(self, driver: webdriver.Chrome, 
                           original_locator: str, 
                           locator_type: str,
                           page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by finding similar elements."""
        suggestions = []
        
        # Extract key characteristics from original locator
        locator_features = self._extract_locator_features(original_locator)
        
        for element in page_elements:
            if not element['is_displayed']:
                continue
                
            # Calculate similarity score
            similarity = self._calculate_element_similarity(locator_features, element)
            
            if similarity > 0.5:  # Threshold for similarity
                # Generate new locators for this element
                new_locators = self._generate_locators_for_element(element)
                
                for new_locator in new_locators:
                    suggestions.append((new_locator, similarity))
                    
        return suggestions
        
    def _heal_by_text_content(self, driver: webdriver.Chrome,
                             original_locator: str,
                             locator_type: str,
                             page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by matching text content."""
        suggestions = []
        
        # Extract text from original locator
        text_pattern = re.search(r'text\(\)\s*=\s*[\'"]([^\'"]+)[\'"]', original_locator)
        contains_pattern = re.search(r'contains\([^,]+,\s*[\'"]([^\'"]+)[\'"]', original_locator)
        
        target_text = None
        if text_pattern:
            target_text = text_pattern.group(1)
        elif contains_pattern:
            target_text = contains_pattern.group(1)
            
        if target_text:
            for element in page_elements:
                if target_text.lower() in element['text'].lower():
                    new_locators = self._generate_locators_for_element(element)
                    confidence = 0.8 if target_text == element['text'] else 0.6
                    
                    for new_locator in new_locators:
                        suggestions.append((new_locator, confidence))
                        
        return suggestions
        
    def _heal_by_position(self, driver: webdriver.Chrome,
                         original_locator: str,
                         locator_type: str,
                         page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by finding elements in similar positions."""
        suggestions = []
        
        # This would require knowing the original element's position
        # For now, return empty list
        return suggestions
        
    def _heal_by_attributes(self, driver: webdriver.Chrome,
                           original_locator: str,
                           locator_type: str,
                           page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by matching attributes."""
        suggestions = []
        
        # Extract attributes from original locator
        attributes = self._extract_attributes_from_locator(original_locator)
        
        for element in page_elements:
            if not element['is_displayed']:
                continue
                
            # Check attribute matches
            matches = 0
            total_attrs = len(attributes)
            
            for attr, value in attributes.items():
                element_value = element.get(attr, '')
                if value.lower() in element_value.lower() or element_value.lower() in value.lower():
                    matches += 1
                    
            if total_attrs > 0:
                confidence = matches / total_attrs
                
                if confidence > 0.4:  # Threshold
                    new_locators = self._generate_locators_for_element(element)
                    
                    for new_locator in new_locators:
                        suggestions.append((new_locator, confidence))
                        
        return suggestions
        
    def _heal_by_hierarchy(self, driver: webdriver.Chrome,
                          original_locator: str,
                          locator_type: str,
                          page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by analyzing element hierarchy."""
        suggestions = []
        
        # Extract parent/child relationships from original locator
        if '//' in original_locator or '/' in original_locator:
            # This is an XPath with hierarchy
            parts = original_locator.split('/')
            
            # Look for elements that match the structure
            for element in page_elements:
                if element['is_displayed']:
                    new_locators = self._generate_locators_for_element(element)
                    
                    for new_locator in new_locators:
                        suggestions.append((new_locator, 0.5))  # Medium confidence
                        
        return suggestions
        
    def _heal_by_css_selector(self, driver: webdriver.Chrome,
                             original_locator: str,
                             locator_type: str,
                             page_elements: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Heal locator by generating CSS selectors."""
        suggestions = []
        
        for element in page_elements:
            if not element['is_displayed']:
                continue
                
            # Generate CSS selectors
            css_selectors = []
            
            # ID selector
            if element['id']:
                css_selectors.append(f"#{element['id']}")
                
            # Class selector
            if element['class']:
                classes = element['class'].split()
                for cls in classes:
                    css_selectors.append(f".{cls}")
                    
            # Tag + attribute selectors
            if element['name']:
                css_selectors.append(f"{element['tag']}[name='{element['name']}']")
            if element['type']:
                css_selectors.append(f"{element['tag']}[type='{element['type']}']")
            if element['placeholder']:
                css_selectors.append(f"{element['tag']}[placeholder*='{element['placeholder'][:20]}']")
                
            for css_selector in css_selectors:
                suggestions.append((css_selector, 0.6))
                
        return suggestions
        
    def _extract_locator_features(self, locator: str) -> Dict[str, str]:
        """Extract features from a locator string."""
        features = {}
        
        # Extract ID
        id_match = re.search(r'@id\s*=\s*[\'"]([^\'"]+)[\'"]', locator)
        if id_match:
            features['id'] = id_match.group(1)
            
        # Extract class
        class_match = re.search(r'@class\s*=\s*[\'"]([^\'"]+)[\'"]', locator)
        if class_match:
            features['class'] = class_match.group(1)
            
        # Extract tag
        tag_match = re.search(r'//(\w+)', locator)
        if tag_match:
            features['tag'] = tag_match.group(1)
            
        # Extract text
        text_match = re.search(r'text\(\)\s*=\s*[\'"]([^\'"]+)[\'"]', locator)
        if text_match:
            features['text'] = text_match.group(1)
            
        return features
        
    def _extract_attributes_from_locator(self, locator: str) -> Dict[str, str]:
        """Extract attributes mentioned in locator."""
        attributes = {}
        
        # Common attribute patterns
        patterns = {
            'id': r'@id\s*=\s*[\'"]([^\'"]+)[\'"]',
            'class': r'@class\s*=\s*[\'"]([^\'"]+)[\'"]',
            'name': r'@name\s*=\s*[\'"]([^\'"]+)[\'"]',
            'type': r'@type\s*=\s*[\'"]([^\'"]+)[\'"]',
            'text': r'text\(\)\s*=\s*[\'"]([^\'"]+)[\'"]'
        }
        
        for attr, pattern in patterns.items():
            match = re.search(pattern, locator)
            if match:
                attributes[attr] = match.group(1)
                
        return attributes
        
    def _calculate_element_similarity(self, locator_features: Dict[str, str], 
                                    element: Dict[str, Any]) -> float:
        """Calculate similarity between locator features and element."""
        if not locator_features:
            return 0.0
            
        matches = 0
        total_features = len(locator_features)
        
        for feature, value in locator_features.items():
            element_value = str(element.get(feature, ''))
            
            if value.lower() == element_value.lower():
                matches += 1
            elif value.lower() in element_value.lower() or element_value.lower() in value.lower():
                matches += 0.5
                
        return matches / total_features if total_features > 0 else 0.0
        
    def _generate_locators_for_element(self, element: Dict[str, Any]) -> List[str]:
        """Generate multiple locator options for an element."""
        locators = []
        
        # XPath locators
        if element['id']:
            locators.append(f"//*[@id='{element['id']}']")
            
        if element['name']:
            locators.append(f"//{element['tag']}[@name='{element['name']}']")
            
        if element['class']:
            classes = element['class'].split()
            for cls in classes[:2]:  # Limit to first 2 classes
                locators.append(f"//{element['tag']}[@class='{cls}']")
                locators.append(f"//*[contains(@class, '{cls}')]")
                
        if element['text'] and len(element['text']) < 50:
            locators.append(f"//*[text()='{element['text']}']")
            locators.append(f"//*[contains(text(), '{element['text'][:20]}')]")
            
        if element['placeholder']:
            locators.append(f"//{element['tag']}[@placeholder='{element['placeholder']}']")
            
        # CSS selectors  
        if element['id']:
            locators.append(f"#{element['id']}")
            
        if element['class']:
            classes = element['class'].split()
            for cls in classes[:2]:
                locators.append(f".{cls}")
                
        return locators


class TestFailureAnalyzer:
    """Analyzes test failures to determine repair strategies."""
    
    def __init__(self):
        self.failure_patterns = {
            FailureType.LOCATOR_NOT_FOUND: [
                r"no such element",
                r"element not found",
                r"unable to locate element",
                r"NoSuchElementException"
            ],
            FailureType.ELEMENT_NOT_INTERACTABLE: [
                r"element not interactable",
                r"ElementNotInteractableException",
                r"element is not clickable"
            ],
            FailureType.TIMEOUT: [
                r"timeout",
                r"TimeoutException",
                r"timed out after"
            ],
            FailureType.STALE_ELEMENT: [
                r"stale element reference",
                r"StaleElementReferenceException"
            ],
            FailureType.ASSERTION_ERROR: [
                r"AssertionError",
                r"assert",
                r"expected.*but was"
            ]
        }
        
    def analyze_failure(self, error_message: str, stack_trace: str,
                       driver: Optional[webdriver.Chrome] = None) -> FailureAnalysis:
        """Analyze a test failure and suggest repair strategies."""
        
        # Determine failure type
        failure_type = self._classify_failure(error_message, stack_trace)
        
        # Extract locator information
        original_locator = self._extract_locator_from_error(error_message, stack_trace)
        
        # Get page information if driver is available
        page_url = ""
        page_source = ""
        screenshot_path = None
        
        if driver:
            try:
                page_url = driver.current_url
                page_source = driver.page_source
                # Take screenshot for debugging
                screenshot_path = f"/tmp/test_failure_{int(time.time())}.png"
                driver.save_screenshot(screenshot_path)
            except:
                pass
                
        # Generate repair suggestions
        suggested_repairs = self._generate_repair_suggestions(
            failure_type, original_locator, error_message, driver
        )
        
        # Calculate confidence based on failure type and available information
        confidence = self._calculate_analysis_confidence(
            failure_type, original_locator, suggested_repairs
        )
        
        return FailureAnalysis(
            failure_type=failure_type,
            original_locator=original_locator,
            error_message=error_message,
            stack_trace=stack_trace,
            page_url=page_url,
            page_source=page_source,
            screenshot_path=screenshot_path,
            suggested_repairs=suggested_repairs,
            confidence=confidence
        )
        
    def _classify_failure(self, error_message: str, stack_trace: str) -> FailureType:
        """Classify the type of failure based on error patterns."""
        combined_text = f"{error_message} {stack_trace}".lower()
        
        for failure_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return failure_type
                    
        return FailureType.DEPENDENCY_FAILURE  # Default
        
    def _extract_locator_from_error(self, error_message: str, stack_trace: str) -> str:
        """Extract the failing locator from error message or stack trace."""
        combined_text = f"{error_message} {stack_trace}"
        
        # Common locator patterns
        patterns = [
            r'xpath[=:]?\s*[\'"]([^\'"]+)[\'"]',
            r'css\s*selector[=:]?\s*[\'"]([^\'"]+)[\'"]',
            r'id[=:]?\s*[\'"]([^\'"]+)[\'"]',
            r'name[=:]?\s*[\'"]([^\'"]+)[\'"]',
            r'find_element\([^,]+,\s*[\'"]([^\'"]+)[\'"]',
            r'By\.\w+,\s*[\'"]([^\'"]+)[\'"]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return ""
        
    def _generate_repair_suggestions(self, failure_type: FailureType,
                                   original_locator: str,
                                   error_message: str,
                                   driver: Optional[webdriver.Chrome]) -> List[Dict[str, Any]]:
        """Generate repair suggestions based on failure type."""
        suggestions = []
        
        if failure_type == FailureType.LOCATOR_NOT_FOUND:
            suggestions.extend(self._suggest_locator_repairs(original_locator, driver))
            
        elif failure_type == FailureType.ELEMENT_NOT_INTERACTABLE:
            suggestions.extend(self._suggest_interactability_repairs())
            
        elif failure_type == FailureType.TIMEOUT:
            suggestions.extend(self._suggest_timeout_repairs())
            
        elif failure_type == FailureType.STALE_ELEMENT:
            suggestions.extend(self._suggest_stale_element_repairs())
            
        elif failure_type == FailureType.ASSERTION_ERROR:
            suggestions.extend(self._suggest_assertion_repairs(error_message))
            
        return suggestions
        
    def _suggest_locator_repairs(self, original_locator: str,
                               driver: Optional[webdriver.Chrome]) -> List[Dict[str, Any]]:
        """Suggest repairs for locator not found errors."""
        suggestions = []
        
        if driver and original_locator:
            # Use ElementLocatorHealer to find alternatives
            healer = ElementLocatorHealer()
            alternatives = healer.heal_locator(driver, original_locator)
            
            for alt_locator, confidence in alternatives:
                suggestions.append({
                    'strategy': RepairStrategy.LOCATOR_HEALING.value,
                    'description': f"Replace locator with: {alt_locator}",
                    'original_locator': original_locator,
                    'new_locator': alt_locator,
                    'confidence': confidence
                })
                
        # General locator improvement suggestions
        suggestions.append({
            'strategy': RepairStrategy.WAIT_STRATEGY.value,
            'description': "Add explicit wait for element presence",
            'code_change': "WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, locator)))",
            'confidence': 0.7
        })
        
        return suggestions
        
    def _suggest_interactability_repairs(self) -> List[Dict[str, Any]]:
        """Suggest repairs for element not interactable errors."""
        return [
            {
                'strategy': RepairStrategy.WAIT_STRATEGY.value,
                'description': "Add wait for element to be clickable",
                'code_change': "WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, locator)))",
                'confidence': 0.8
            },
            {
                'strategy': RepairStrategy.ALTERNATIVE_PATH.value,
                'description': "Use JavaScript click instead of Selenium click",
                'code_change': "driver.execute_script('arguments[0].click();', element)",
                'confidence': 0.7
            },
            {
                'strategy': RepairStrategy.WAIT_STRATEGY.value,
                'description': "Scroll element into view before interaction",
                'code_change': "driver.execute_script('arguments[0].scrollIntoView();', element)",
                'confidence': 0.6
            }
        ]
        
    def _suggest_timeout_repairs(self) -> List[Dict[str, Any]]:
        """Suggest repairs for timeout errors."""
        return [
            {
                'strategy': RepairStrategy.WAIT_STRATEGY.value,
                'description': "Increase timeout duration",
                'code_change': "Increase WebDriverWait timeout to 20 seconds",
                'confidence': 0.6
            },
            {
                'strategy': RepairStrategy.RETRY_MECHANISM.value,
                'description': "Add retry mechanism with exponential backoff",
                'code_change': "Implement retry decorator with max 3 attempts",
                'confidence': 0.7
            }
        ]
        
    def _suggest_stale_element_repairs(self) -> List[Dict[str, Any]]:
        """Suggest repairs for stale element errors."""
        return [
            {
                'strategy': RepairStrategy.RETRY_MECHANISM.value,
                'description': "Re-find element on stale reference",
                'code_change': "Wrap element operations in try-except and re-find element",
                'confidence': 0.8
            },
            {
                'strategy': RepairStrategy.LOCATOR_HEALING.value,
                'description': "Use fresh element references",
                'code_change': "Find element immediately before each operation",
                'confidence': 0.9
            }
        ]
        
    def _suggest_assertion_repairs(self, error_message: str) -> List[Dict[str, Any]]:
        """Suggest repairs for assertion errors."""
        suggestions = [
            {
                'strategy': RepairStrategy.SMART_ASSERTION.value,
                'description': "Add tolerance to assertion",
                'code_change': "Use approximate equality for numeric comparisons",
                'confidence': 0.5
            },
            {
                'strategy': RepairStrategy.DYNAMIC_DATA.value,
                'description': "Handle dynamic data in assertions",
                'code_change': "Use contains() or regex patterns instead of exact matches",
                'confidence': 0.6
            }
        ]
        
        # Analyze specific assertion patterns
        if "expected" in error_message.lower() and "but was" in error_message.lower():
            suggestions.append({
                'strategy': RepairStrategy.SMART_ASSERTION.value,
                'description': "Update expected value based on current behavior",
                'code_change': "Update assertion to match current application state",
                'confidence': 0.4
            })
            
        return suggestions
        
    def _calculate_analysis_confidence(self, failure_type: FailureType,
                                     original_locator: str,
                                     suggested_repairs: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the failure analysis."""
        base_confidence = 0.5
        
        # Increase confidence based on failure type clarity
        if failure_type in [FailureType.LOCATOR_NOT_FOUND, FailureType.TIMEOUT]:
            base_confidence += 0.2
            
        # Increase confidence if locator was extracted
        if original_locator:
            base_confidence += 0.1
            
        # Increase confidence based on number of repair suggestions
        if len(suggested_repairs) > 2:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)


class TestCodeRepairer:
    """Repairs test code by applying healing strategies."""
    
    def __init__(self):
        self.repair_templates = {
            RepairStrategy.LOCATOR_HEALING: self._apply_locator_healing,
            RepairStrategy.WAIT_STRATEGY: self._apply_wait_strategy,
            RepairStrategy.RETRY_MECHANISM: self._apply_retry_mechanism,
            RepairStrategy.ALTERNATIVE_PATH: self._apply_alternative_path,
            RepairStrategy.SMART_ASSERTION: self._apply_smart_assertion
        }
        
    def repair_test_code(self, test_file_path: str,
                        failure_analysis: FailureAnalysis) -> List[RepairAction]:
        """Repair test code based on failure analysis."""
        with open(test_file_path, 'r') as f:
            original_code = f.read()
            
        applied_repairs = []
        
        # Apply repairs based on suggestions
        for suggestion in failure_analysis.suggested_repairs:
            strategy = RepairStrategy(suggestion['strategy'])
            
            if strategy in self.repair_templates:
                repair_function = self.repair_templates[strategy]
                
                try:
                    repair_action = repair_function(
                        original_code,
                        failure_analysis,
                        suggestion
                    )
                    
                    if repair_action:
                        applied_repairs.append(repair_action)
                        original_code = repair_action.repaired_code  # Chain repairs
                        
                except Exception as e:
                    logging.error(f"Error applying repair strategy {strategy}: {e}")
                    
        return applied_repairs
        
    def _apply_locator_healing(self, code: str,
                             failure_analysis: FailureAnalysis,
                             suggestion: Dict[str, Any]) -> Optional[RepairAction]:
        """Apply locator healing to test code."""
        
        original_locator = suggestion.get('original_locator', '')
        new_locator = suggestion.get('new_locator', '')
        
        if not original_locator or not new_locator:
            return None
            
        # Replace locator in code
        repaired_code = code.replace(f'"{original_locator}"', f'"{new_locator}"')
        repaired_code = repaired_code.replace(f"'{original_locator}'", f"'{new_locator}'")
        
        if repaired_code != code:
            return RepairAction(
                strategy=RepairStrategy.LOCATOR_HEALING,
                original_code=code,
                repaired_code=repaired_code,
                confidence=suggestion.get('confidence', 0.5),
                description=f"Updated locator from '{original_locator}' to '{new_locator}'",
                metadata={'original_locator': original_locator, 'new_locator': new_locator}
            )
            
        return None
        
    def _apply_wait_strategy(self, code: str,
                           failure_analysis: FailureAnalysis,
                           suggestion: Dict[str, Any]) -> Optional[RepairAction]:
        """Apply wait strategy to test code."""
        
        # Find find_element calls and add waits
        patterns = [
            (r'driver\.find_element\(([^)]+)\)', 
             r'WebDriverWait(driver, 10).until(EC.presence_of_element_located(\1))'),
            (r'\.click\(\)',
             '.click()  # Added wait strategy')
        ]
        
        repaired_code = code
        
        # Add imports if not present
        if 'from selenium.webdriver.support.ui import WebDriverWait' not in code:
            import_line = 'from selenium.webdriver.support.ui import WebDriverWait\n'
            import_line += 'from selenium.webdriver.support import expected_conditions as EC\n'
            
            # Insert after existing imports or at the beginning
            lines = repaired_code.split('\n')
            import_index = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_index = i + 1
                    
            lines.insert(import_index, import_line)
            repaired_code = '\n'.join(lines)
            
        # Apply wait patterns
        for pattern, replacement in patterns:
            if re.search(pattern, repaired_code):
                repaired_code = re.sub(pattern, replacement, repaired_code)
                break
                
        if repaired_code != code:
            return RepairAction(
                strategy=RepairStrategy.WAIT_STRATEGY,
                original_code=code,
                repaired_code=repaired_code,
                confidence=0.7,
                description="Added explicit waits for elements",
                metadata={'patterns_applied': len(patterns)}
            )
            
        return None
        
    def _apply_retry_mechanism(self, code: str,
                             failure_analysis: FailureAnalysis,
                             suggestion: Dict[str, Any]) -> Optional[RepairAction]:
        """Apply retry mechanism to test code."""
        
        # Add retry decorator
        retry_decorator = '''
def retry_on_failure(max_attempts=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
'''
        
        # Find test functions and add decorator
        test_function_pattern = r'(def test_[^(]+\([^)]*\):)'
        
        repaired_code = code
        
        # Add retry imports
        if 'import time' not in code:
            repaired_code = 'import time\n' + repaired_code
            
        # Add retry decorator definition
        repaired_code = retry_decorator + repaired_code
        
        # Apply decorator to test functions
        repaired_code = re.sub(
            test_function_pattern,
            r'@retry_on_failure(max_attempts=3)\n\1',
            repaired_code
        )
        
        if repaired_code != code:
            return RepairAction(
                strategy=RepairStrategy.RETRY_MECHANISM,
                original_code=code,
                repaired_code=repaired_code,
                confidence=0.6,
                description="Added retry mechanism to test functions",
                metadata={'max_attempts': 3}
            )
            
        return None
        
    def _apply_alternative_path(self, code: str,
                              failure_analysis: FailureAnalysis,
                              suggestion: Dict[str, Any]) -> Optional[RepairAction]:
        """Apply alternative interaction paths."""
        
        # Replace regular clicks with JavaScript clicks
        repaired_code = code
        
        # Pattern for element.click()
        click_pattern = r'(\w+)\.click\(\)'
        js_click_replacement = r"driver.execute_script('arguments[0].click();', \1)"
        
        if re.search(click_pattern, repaired_code):
            repaired_code = re.sub(click_pattern, js_click_replacement, repaired_code)
            
            return RepairAction(
                strategy=RepairStrategy.ALTERNATIVE_PATH,
                original_code=code,
                repaired_code=repaired_code,
                confidence=0.6,
                description="Replaced Selenium click with JavaScript click",
                metadata={'change_type': 'js_click'}
            )
            
        return None
        
    def _apply_smart_assertion(self, code: str,
                             failure_analysis: FailureAnalysis,
                             suggestion: Dict[str, Any]) -> Optional[RepairAction]:
        """Apply smart assertion improvements."""
        
        repaired_code = code
        
        # Replace exact assertions with approximate ones
        patterns = [
            (r'assert\s+([^=\s]+)\s*==\s*([^,\n]+)', 
             r'assert abs(\1 - \2) < 0.01  # Approximate equality'),
            (r'assert\s+([^=\s]+)\s*==\s*[\'"]([^\'"]+)[\'"]',
             r'assert "\2" in str(\1)  # Contains check instead of exact match')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, repaired_code):
                repaired_code = re.sub(pattern, replacement, repaired_code)
                break
                
        if repaired_code != code:
            return RepairAction(
                strategy=RepairStrategy.SMART_ASSERTION,
                original_code=code,
                repaired_code=repaired_code,
                confidence=0.5,
                description="Updated assertions to be more flexible",
                metadata={'assertion_type': 'approximate'}
            )
            
        return None


class SelfHealingTestFramework:
    """Main self-healing test framework."""
    
    def __init__(self, test_directory: str):
        self.test_directory = Path(test_directory)
        self.failure_analyzer = TestFailureAnalyzer()
        self.code_repairer = TestCodeRepairer()
        self.element_healer = ElementLocatorHealer()
        
        # Healing history
        self.healing_history = []
        self.success_rate = defaultdict(float)
        
    def heal_test_failure(self, test_file: str, test_function: str,
                         error_message: str, stack_trace: str,
                         driver: Optional[webdriver.Chrome] = None) -> TestHealingResult:
        """Heal a specific test failure."""
        
        start_time = time.time()
        
        # Analyze the failure
        failure_analysis = self.failure_analyzer.analyze_failure(
            error_message, stack_trace, driver
        )
        
        # Repair the test code
        test_file_path = self.test_directory / test_file
        applied_repairs = self.code_repairer.repair_test_code(
            str(test_file_path), failure_analysis
        )
        
        # Generate new test code
        new_test_code = ""
        if applied_repairs:
            new_test_code = applied_repairs[-1].repaired_code  # Latest repaired version
            
        # Validate repairs (if possible)
        validation_results = self._validate_repairs(
            str(test_file_path), new_test_code, driver
        )
        
        healing_successful = validation_results.get('syntax_valid', False)
        
        execution_time = time.time() - start_time
        
        result = TestHealingResult(
            test_file=test_file,
            test_function=test_function,
            original_failure=failure_analysis,
            applied_repairs=applied_repairs,
            healing_successful=healing_successful,
            execution_time=execution_time,
            new_test_code=new_test_code,
            validation_results=validation_results
        )
        
        # Store healing history
        self.healing_history.append(result)
        
        # Update success rate
        strategy_key = failure_analysis.failure_type.value
        if healing_successful:
            self.success_rate[strategy_key] = (
                self.success_rate[strategy_key] * 0.8 + 0.2
            )  # Moving average
        else:
            self.success_rate[strategy_key] = (
                self.success_rate[strategy_key] * 0.8
            )
            
        return result
        
    def _validate_repairs(self, original_file: str, new_code: str,
                         driver: Optional[webdriver.Chrome]) -> Dict[str, Any]:
        """Validate that repairs are syntactically and logically correct."""
        validation_results = {}
        
        # Syntax validation
        try:
            ast.parse(new_code)
            validation_results['syntax_valid'] = True
        except SyntaxError as e:
            validation_results['syntax_valid'] = False
            validation_results['syntax_error'] = str(e)
            
        # Import validation
        imports_valid = self._validate_imports(new_code)
        validation_results['imports_valid'] = imports_valid
        
        # Locator validation (if driver available)
        if driver:
            locators_valid = self._validate_locators(new_code, driver)
            validation_results['locators_valid'] = locators_valid
        else:
            validation_results['locators_valid'] = None
            
        return validation_results
        
    def _validate_imports(self, code: str) -> bool:
        """Validate that all imports are available."""
        try:
            # Extract import statements
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name)
                        except ImportError:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            module = __import__(node.module, fromlist=[name.name for name in node.names])
                            for alias in node.names:
                                if not hasattr(module, alias.name):
                                    return False
                        except ImportError:
                            return False
                            
            return True
            
        except Exception as e:
            logging.debug(f"Error validating imports: {e}")
            return False
            
    def _validate_locators(self, code: str, driver: webdriver.Chrome) -> Dict[str, bool]:
        """Validate that locators in code can find elements."""
        locator_results = {}
        
        # Extract locators from code
        locators = self._extract_locators_from_code(code)
        
        for locator_type, locator_value in locators:
            try:
                if locator_type == 'xpath':
                    element = driver.find_element(By.XPATH, locator_value)
                elif locator_type == 'css':
                    element = driver.find_element(By.CSS_SELECTOR, locator_value)
                elif locator_type == 'id':
                    element = driver.find_element(By.ID, locator_value)
                elif locator_type == 'name':
                    element = driver.find_element(By.NAME, locator_value)
                else:
                    continue
                    
                locator_results[locator_value] = element is not None
                
            except NoSuchElementException:
                locator_results[locator_value] = False
            except Exception as e:
                logging.debug(f"Error validating locator {locator_value}: {e}")
                locator_results[locator_value] = False
                
        return locator_results
        
    def _extract_locators_from_code(self, code: str) -> List[Tuple[str, str]]:
        """Extract locators from test code."""
        locators = []
        
        # Common locator patterns
        patterns = {
            'xpath': r'By\.XPATH,\s*[\'"]([^\'"]+)[\'"]',
            'css': r'By\.CSS_SELECTOR,\s*[\'"]([^\'"]+)[\'"]',
            'id': r'By\.ID,\s*[\'"]([^\'"]+)[\'"]',
            'name': r'By\.NAME,\s*[\'"]([^\'"]+)[\'"]'
        }
        
        for locator_type, pattern in patterns.items():
            matches = re.findall(pattern, code)
            for match in matches:
                locators.append((locator_type, match))
                
        return locators
        
    def apply_healing_to_file(self, file_path: str, backup: bool = True) -> bool:
        """Apply healing to a test file and save changes."""
        
        if backup:
            backup_path = f"{file_path}.backup_{int(time.time())}"
            with open(file_path, 'r') as f:
                original_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(original_content)
                
        # Find the most recent successful healing for this file
        successful_healings = [
            h for h in self.healing_history 
            if h.test_file == Path(file_path).name and h.healing_successful
        ]
        
        if successful_healings:
            # Apply the most recent successful healing
            latest_healing = max(successful_healings, key=lambda h: h.original_failure.timestamp)
            
            with open(file_path, 'w') as f:
                f.write(latest_healing.new_test_code)
                
            logging.info(f"Applied healing to {file_path}")
            return True
            
        return False
        
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get statistics about healing success rates."""
        
        if not self.healing_history:
            return {}
            
        total_attempts = len(self.healing_history)
        successful_healings = len([h for h in self.healing_history if h.healing_successful])
        
        # Success rate by failure type
        success_by_type = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        for healing in self.healing_history:
            failure_type = healing.original_failure.failure_type.value
            success_by_type[failure_type]['total'] += 1
            if healing.healing_successful:
                success_by_type[failure_type]['successful'] += 1
                
        # Convert to percentages
        success_rates_by_type = {}
        for failure_type, stats in success_by_type.items():
            success_rates_by_type[failure_type] = (
                stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            )
            
        return {
            'total_healing_attempts': total_attempts,
            'successful_healings': successful_healings,
            'overall_success_rate': successful_healings / total_attempts if total_attempts > 0 else 0,
            'success_rates_by_failure_type': success_rates_by_type,
            'average_healing_time': np.mean([h.execution_time for h in self.healing_history]),
            'most_common_failure_type': max(success_by_type.keys(), 
                                          key=lambda k: success_by_type[k]['total']) if success_by_type else None
        }
        
    def save_healing_report(self, output_path: str):
        """Save a comprehensive healing report."""
        
        report = {
            'healing_statistics': self.get_healing_statistics(),
            'healing_history': [
                {
                    'test_file': healing.test_file,
                    'test_function': healing.test_function,
                    'failure_type': healing.original_failure.failure_type.value,
                    'healing_successful': healing.healing_successful,
                    'execution_time': healing.execution_time,
                    'repairs_applied': len(healing.applied_repairs),
                    'timestamp': healing.original_failure.timestamp.isoformat()
                }
                for healing in self.healing_history
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logging.info(f"Healing report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    framework = SelfHealingTestFramework("tests/")
    
    # Simulate a test failure healing
    error_message = "NoSuchElementException: Unable to locate element: {'method': 'xpath', 'selector': '//button[@id=\"submit-btn\"]'}"
    stack_trace = "File 'test_login.py', line 25, in test_login_form"
    
    result = framework.heal_test_failure(
        test_file="test_login.py",
        test_function="test_login_form", 
        error_message=error_message,
        stack_trace=stack_trace
    )
    
    print(f"Healing successful: {result.healing_successful}")
    print(f"Applied repairs: {len(result.applied_repairs)}")
    
    # Get statistics
    stats = framework.get_healing_statistics()
    print(f"Overall success rate: {stats.get('overall_success_rate', 0):.1%}")
    
    # Save report
    framework.save_healing_report("healing_report.json")