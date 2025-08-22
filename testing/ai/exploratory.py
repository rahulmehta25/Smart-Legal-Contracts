"""
AI-Powered Exploratory Testing

This module provides intelligent exploratory testing that automatically navigates
applications, discovers functionality, and identifies potential issues using AI.
"""

import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import threading
from urllib.parse import urljoin, urlparse
import hashlib
import re

import numpy as np
import networkx as nx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, ElementNotInteractableException,
    StaleElementReferenceException, WebDriverException
)
import requests
from bs4 import BeautifulSoup


class ExplorationStrategy(Enum):
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RANDOM = "random"
    SMART = "smart"
    MODEL_BASED = "model_based"


class InteractionType(Enum):
    CLICK = "click"
    INPUT = "input"
    SELECT = "select"
    HOVER = "hover"
    SCROLL = "scroll"
    KEYBOARD = "keyboard"
    FORM_SUBMIT = "form_submit"
    DRAG_DROP = "drag_drop"


class IssueType(Enum):
    BROKEN_LINK = "broken_link"
    JS_ERROR = "js_error"
    LAYOUT_ISSUE = "layout_issue"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    FUNCTIONAL = "functional"


@dataclass
class PageState:
    """Represents the state of a web page."""
    url: str
    title: str
    dom_hash: str
    interactive_elements: List[Dict[str, Any]]
    forms: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    errors: List[str]
    performance_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExplorationAction:
    """Represents an action taken during exploration."""
    action_type: InteractionType
    element_info: Dict[str, Any]
    input_data: Optional[str]
    result: str  # 'success', 'failure', 'error'
    before_state: str
    after_state: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiscoveredIssue:
    """Represents an issue discovered during exploration."""
    issue_type: IssueType
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    location: Dict[str, Any]
    reproduction_steps: List[ExplorationAction]
    evidence: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class WebPageAnalyzer:
    """Analyzes web pages to extract actionable elements and information."""
    
    def __init__(self):
        self.known_patterns = {
            'login_forms': ['login', 'signin', 'sign-in', 'auth'],
            'search_boxes': ['search', 'query', 'find'],
            'navigation': ['nav', 'menu', 'header', 'sidebar'],
            'buttons': ['button', 'btn', 'submit', 'save', 'cancel'],
            'forms': ['form', 'input', 'textarea', 'select']
        }
        
    def analyze_page(self, driver: webdriver.Chrome) -> PageState:
        """Analyze current page state."""
        try:
            # Basic page information
            url = driver.current_url
            title = driver.title
            
            # Get DOM hash for state comparison
            dom_content = driver.page_source
            dom_hash = hashlib.md5(dom_content.encode()).hexdigest()
            
            # Find interactive elements
            interactive_elements = self._find_interactive_elements(driver)
            
            # Find forms
            forms = self._find_forms(driver)
            
            # Find links
            links = self._find_links(driver)
            
            # Check for JavaScript errors
            errors = self._get_console_errors(driver)
            
            # Get performance metrics
            performance_metrics = self._get_performance_metrics(driver)
            
            return PageState(
                url=url,
                title=title,
                dom_hash=dom_hash,
                interactive_elements=interactive_elements,
                forms=forms,
                links=links,
                errors=errors,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logging.error(f"Error analyzing page: {e}")
            return PageState("", "", "", [], [], [], [str(e)], {})
            
    def _find_interactive_elements(self, driver: webdriver.Chrome) -> List[Dict[str, Any]]:
        """Find all interactive elements on the page."""
        elements = []
        
        # Common interactive element selectors
        selectors = [
            "button", "input", "select", "textarea", "a",
            "[onclick]", "[role='button']", ".btn", ".button",
            "[data-action]", "[href]", "[type='submit']"
        ]
        
        for selector in selectors:
            try:
                web_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                for elem in web_elements:
                    if self._is_element_interactable(elem):
                        element_info = self._extract_element_info(elem)
                        if element_info:
                            elements.append(element_info)
                            
            except Exception as e:
                logging.debug(f"Error finding elements with selector {selector}: {e}")
                
        return elements
        
    def _find_forms(self, driver: webdriver.Chrome) -> List[Dict[str, Any]]:
        """Find all forms on the page."""
        forms = []
        
        try:
            form_elements = driver.find_elements(By.TAG_NAME, "form")
            
            for form in form_elements:
                form_info = {
                    'action': form.get_attribute('action') or '',
                    'method': form.get_attribute('method') or 'get',
                    'id': form.get_attribute('id') or '',
                    'class': form.get_attribute('class') or '',
                    'inputs': []
                }
                
                # Find inputs within form
                inputs = form.find_elements(By.CSS_SELECTOR, "input, textarea, select")
                for input_elem in inputs:
                    input_info = {
                        'type': input_elem.get_attribute('type') or 'text',
                        'name': input_elem.get_attribute('name') or '',
                        'id': input_elem.get_attribute('id') or '',
                        'placeholder': input_elem.get_attribute('placeholder') or '',
                        'required': input_elem.get_attribute('required') is not None,
                        'xpath': self._get_xpath(driver, input_elem)
                    }
                    form_info['inputs'].append(input_info)
                    
                form_info['xpath'] = self._get_xpath(driver, form)
                forms.append(form_info)
                
        except Exception as e:
            logging.error(f"Error finding forms: {e}")
            
        return forms
        
    def _find_links(self, driver: webdriver.Chrome) -> List[Dict[str, Any]]:
        """Find all links on the page."""
        links = []
        
        try:
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            
            for link in link_elements:
                href = link.get_attribute('href')
                if href and href.startswith(('http', '/')):
                    link_info = {
                        'href': href,
                        'text': link.text.strip(),
                        'title': link.get_attribute('title') or '',
                        'target': link.get_attribute('target') or '_self',
                        'xpath': self._get_xpath(driver, link)
                    }
                    links.append(link_info)
                    
        except Exception as e:
            logging.error(f"Error finding links: {e}")
            
        return links
        
    def _is_element_interactable(self, element) -> bool:
        """Check if an element is interactable."""
        try:
            return (element.is_displayed() and 
                   element.is_enabled() and 
                   element.size['height'] > 0 and 
                   element.size['width'] > 0)
        except:
            return False
            
    def _extract_element_info(self, element) -> Optional[Dict[str, Any]]:
        """Extract information from an element."""
        try:
            return {
                'tag': element.tag_name,
                'type': element.get_attribute('type') or '',
                'id': element.get_attribute('id') or '',
                'class': element.get_attribute('class') or '',
                'text': element.text.strip(),
                'placeholder': element.get_attribute('placeholder') or '',
                'href': element.get_attribute('href') or '',
                'onclick': element.get_attribute('onclick') or '',
                'location': element.location,
                'size': element.size,
                'xpath': self._get_xpath(element.parent, element) if hasattr(element, 'parent') else ''
            }
        except Exception as e:
            logging.debug(f"Error extracting element info: {e}")
            return None
            
    def _get_xpath(self, driver, element) -> str:
        """Get XPath for an element."""
        try:
            return driver.execute_script(
                "return arguments[0].getXPathNode ? arguments[0].getXPathNode() : '';",
                element
            ) or ''
        except:
            return ''
            
    def _get_console_errors(self, driver: webdriver.Chrome) -> List[str]:
        """Get JavaScript console errors."""
        try:
            logs = driver.get_log('browser')
            errors = []
            
            for entry in logs:
                if entry['level'] in ['SEVERE', 'ERROR']:
                    errors.append(entry['message'])
                    
            return errors
        except Exception as e:
            logging.debug(f"Error getting console errors: {e}")
            return []
            
    def _get_performance_metrics(self, driver: webdriver.Chrome) -> Dict[str, float]:
        """Get page performance metrics."""
        try:
            # Get navigation timing
            timing = driver.execute_script("return performance.timing")
            
            if timing:
                return {
                    'load_time': timing.get('loadEventEnd', 0) - timing.get('navigationStart', 0),
                    'dom_ready': timing.get('domContentLoadedEventEnd', 0) - timing.get('navigationStart', 0),
                    'first_paint': timing.get('responseStart', 0) - timing.get('navigationStart', 0)
                }
        except Exception as e:
            logging.debug(f"Error getting performance metrics: {e}")
            
        return {'load_time': 0, 'dom_ready': 0, 'first_paint': 0}


class IntelligentInputGenerator:
    """Generates intelligent test inputs based on field types and context."""
    
    def __init__(self):
        self.input_patterns = {
            'email': ['test@example.com', 'user@test.com', 'admin@site.com'],
            'password': ['Test123!', 'password123', 'SecurePass!'],
            'username': ['testuser', 'admin', 'user123'],
            'name': ['John Doe', 'Test User', 'Jane Smith'],
            'phone': ['555-0123', '(555) 123-4567', '+1-555-123-4567'],
            'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'city': ['New York', 'Los Angeles', 'Chicago'],
            'zip': ['12345', '90210', '10001'],
            'url': ['http://example.com', 'https://test.com', 'www.site.com'],
            'number': ['123', '456', '789'],
            'date': ['2023-01-01', '2023-12-31', '2024-06-15'],
            'search': ['test query', 'search term', 'example'],
            'default': ['test input', 'sample data', 'example text']
        }
        
    def generate_input(self, element_info: Dict[str, Any]) -> str:
        """Generate appropriate input for an element."""
        element_type = element_info.get('type', '').lower()
        element_name = element_info.get('name', '').lower()
        element_id = element_info.get('id', '').lower()
        placeholder = element_info.get('placeholder', '').lower()
        
        # Combine all text for pattern matching
        combined_text = f"{element_type} {element_name} {element_id} {placeholder}"
        
        # Match against known patterns
        for pattern, values in self.input_patterns.items():
            if pattern in combined_text:
                return random.choice(values)
                
        # Type-specific generation
        if element_type == 'email':
            return random.choice(self.input_patterns['email'])
        elif element_type == 'password':
            return random.choice(self.input_patterns['password'])
        elif element_type == 'tel':
            return random.choice(self.input_patterns['phone'])
        elif element_type == 'url':
            return random.choice(self.input_patterns['url'])
        elif element_type == 'number':
            return random.choice(self.input_patterns['number'])
        elif element_type == 'date':
            return random.choice(self.input_patterns['date'])
        elif element_type == 'search':
            return random.choice(self.input_patterns['search'])
        
        # Context-based generation
        if any(word in combined_text for word in ['email', 'mail']):
            return random.choice(self.input_patterns['email'])
        elif any(word in combined_text for word in ['pass', 'pwd']):
            return random.choice(self.input_patterns['password'])
        elif any(word in combined_text for word in ['user', 'login']):
            return random.choice(self.input_patterns['username'])
        elif any(word in combined_text for word in ['name', 'full']):
            return random.choice(self.input_patterns['name'])
        elif any(word in combined_text for word in ['phone', 'mobile', 'tel']):
            return random.choice(self.input_patterns['phone'])
        elif any(word in combined_text for word in ['address', 'street']):
            return random.choice(self.input_patterns['address'])
        elif any(word in combined_text for word in ['city', 'town']):
            return random.choice(self.input_patterns['city'])
        elif any(word in combined_text for word in ['zip', 'postal']):
            return random.choice(self.input_patterns['zip'])
        elif any(word in combined_text for word in ['search', 'query']):
            return random.choice(self.input_patterns['search'])
        
        # Default fallback
        return random.choice(self.input_patterns['default'])


class IssueDetector:
    """Detects various types of issues during exploration."""
    
    def __init__(self):
        self.detectors = {
            IssueType.BROKEN_LINK: self._detect_broken_links,
            IssueType.JS_ERROR: self._detect_js_errors,
            IssueType.LAYOUT_ISSUE: self._detect_layout_issues,
            IssueType.ACCESSIBILITY: self._detect_accessibility_issues,
            IssueType.PERFORMANCE: self._detect_performance_issues,
            IssueType.SECURITY: self._detect_security_issues,
            IssueType.USABILITY: self._detect_usability_issues,
            IssueType.FUNCTIONAL: self._detect_functional_issues
        }
        
    def detect_issues(self, driver: webdriver.Chrome, 
                     page_state: PageState) -> List[DiscoveredIssue]:
        """Detect all types of issues on the current page."""
        issues = []
        
        for issue_type, detector in self.detectors.items():
            try:
                detected = detector(driver, page_state)
                issues.extend(detected)
            except Exception as e:
                logging.error(f"Error in {issue_type.value} detection: {e}")
                
        return issues
        
    def _detect_broken_links(self, driver: webdriver.Chrome, 
                           page_state: PageState) -> List[DiscoveredIssue]:
        """Detect broken links."""
        issues = []
        
        for link in page_state.links:
            href = link['href']
            
            # Skip external links for now (can be enabled)
            if not href.startswith(('http', '/')):
                continue
                
            try:
                # Make absolute URL
                if href.startswith('/'):
                    base_url = f"{urlparse(page_state.url).scheme}://{urlparse(page_state.url).netloc}"
                    full_url = urljoin(base_url, href)
                else:
                    full_url = href
                    
                # Check link
                response = requests.head(full_url, timeout=5)
                
                if response.status_code >= 400:
                    issues.append(DiscoveredIssue(
                        issue_type=IssueType.BROKEN_LINK,
                        severity='medium',
                        description=f"Broken link: {href} returns {response.status_code}",
                        location={'url': page_state.url, 'link': href},
                        reproduction_steps=[],
                        evidence={'status_code': response.status_code, 'url': full_url},
                        confidence=0.9
                    ))
                    
            except requests.RequestException:
                # Link might be broken or external
                issues.append(DiscoveredIssue(
                    issue_type=IssueType.BROKEN_LINK,
                    severity='low',
                    description=f"Potentially broken link: {href}",
                    location={'url': page_state.url, 'link': href},
                    reproduction_steps=[],
                    evidence={'url': href, 'error': 'Request failed'},
                    confidence=0.5
                ))
                
        return issues
        
    def _detect_js_errors(self, driver: webdriver.Chrome, 
                         page_state: PageState) -> List[DiscoveredIssue]:
        """Detect JavaScript errors."""
        issues = []
        
        for error in page_state.errors:
            severity = 'high' if 'error' in error.lower() else 'medium'
            
            issues.append(DiscoveredIssue(
                issue_type=IssueType.JS_ERROR,
                severity=severity,
                description=f"JavaScript error: {error}",
                location={'url': page_state.url},
                reproduction_steps=[],
                evidence={'error_message': error},
                confidence=0.8
            ))
            
        return issues
        
    def _detect_layout_issues(self, driver: webdriver.Chrome, 
                            page_state: PageState) -> List[DiscoveredIssue]:
        """Detect layout issues."""
        issues = []
        
        try:
            # Check for elements outside viewport
            viewport_width = driver.execute_script("return window.innerWidth")
            viewport_height = driver.execute_script("return window.innerHeight")
            
            elements = driver.find_elements(By.CSS_SELECTOR, "*")
            
            for element in elements[:50]:  # Limit to first 50 elements
                try:
                    location = element.location
                    size = element.size
                    
                    # Check if element is outside viewport
                    if (location['x'] + size['width'] > viewport_width + 50 or
                        location['y'] + size['height'] > viewport_height + 50):
                        
                        # Skip hidden elements
                        if not element.is_displayed():
                            continue
                            
                        issues.append(DiscoveredIssue(
                            issue_type=IssueType.LAYOUT_ISSUE,
                            severity='low',
                            description="Element extends outside viewport",
                            location={'url': page_state.url, 'element': element.tag_name},
                            reproduction_steps=[],
                            evidence={
                                'element_location': location,
                                'element_size': size,
                                'viewport_size': {'width': viewport_width, 'height': viewport_height}
                            },
                            confidence=0.6
                        ))
                        
                except StaleElementReferenceException:
                    continue
                    
        except Exception as e:
            logging.debug(f"Error in layout detection: {e}")
            
        return issues
        
    def _detect_accessibility_issues(self, driver: webdriver.Chrome, 
                                   page_state: PageState) -> List[DiscoveredIssue]:
        """Detect accessibility issues."""
        issues = []
        
        try:
            # Check for images without alt text
            images = driver.find_elements(By.TAG_NAME, "img")
            for img in images:
                alt_text = img.get_attribute("alt")
                if not alt_text:
                    issues.append(DiscoveredIssue(
                        issue_type=IssueType.ACCESSIBILITY,
                        severity='medium',
                        description="Image missing alt text",
                        location={'url': page_state.url, 'element': 'img'},
                        reproduction_steps=[],
                        evidence={'src': img.get_attribute('src')},
                        confidence=0.9
                    ))
                    
            # Check for form inputs without labels
            inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='email'], input[type='password'], textarea")
            for inp in inputs:
                input_id = inp.get_attribute("id")
                if input_id:
                    # Check if there's a label for this input
                    labels = driver.find_elements(By.CSS_SELECTOR, f"label[for='{input_id}']")
                    if not labels and not inp.get_attribute("aria-label") and not inp.get_attribute("placeholder"):
                        issues.append(DiscoveredIssue(
                            issue_type=IssueType.ACCESSIBILITY,
                            severity='high',
                            description="Form input missing label or aria-label",
                            location={'url': page_state.url, 'element': 'input'},
                            reproduction_steps=[],
                            evidence={'input_id': input_id, 'input_type': inp.get_attribute('type')},
                            confidence=0.8
                        ))
                        
        except Exception as e:
            logging.debug(f"Error in accessibility detection: {e}")
            
        return issues
        
    def _detect_performance_issues(self, driver: webdriver.Chrome, 
                                 page_state: PageState) -> List[DiscoveredIssue]:
        """Detect performance issues."""
        issues = []
        
        load_time = page_state.performance_metrics.get('load_time', 0)
        
        if load_time > 5000:  # 5 seconds
            severity = 'high' if load_time > 10000 else 'medium'
            issues.append(DiscoveredIssue(
                issue_type=IssueType.PERFORMANCE,
                severity=severity,
                description=f"Slow page load time: {load_time}ms",
                location={'url': page_state.url},
                reproduction_steps=[],
                evidence={'load_time': load_time},
                confidence=0.9
            ))
            
        return issues
        
    def _detect_security_issues(self, driver: webdriver.Chrome, 
                              page_state: PageState) -> List[DiscoveredIssue]:
        """Detect security issues."""
        issues = []
        
        # Check for forms without HTTPS
        if not page_state.url.startswith('https://'):
            for form in page_state.forms:
                # Check if form has password fields
                has_password = any(inp.get('type') == 'password' for inp in form.get('inputs', []))
                
                if has_password:
                    issues.append(DiscoveredIssue(
                        issue_type=IssueType.SECURITY,
                        severity='critical',
                        description="Password form on non-HTTPS page",
                        location={'url': page_state.url},
                        reproduction_steps=[],
                        evidence={'form_action': form.get('action', '')},
                        confidence=0.9
                    ))
                    
        return issues
        
    def _detect_usability_issues(self, driver: webdriver.Chrome, 
                               page_state: PageState) -> List[DiscoveredIssue]:
        """Detect usability issues."""
        issues = []
        
        # Check for very small clickable elements
        for element_info in page_state.interactive_elements:
            size = element_info.get('size', {})
            if size.get('width', 0) < 44 or size.get('height', 0) < 44:
                issues.append(DiscoveredIssue(
                    issue_type=IssueType.USABILITY,
                    severity='low',
                    description="Clickable element smaller than recommended 44px",
                    location={'url': page_state.url, 'element': element_info.get('tag')},
                    reproduction_steps=[],
                    evidence={'size': size},
                    confidence=0.7
                ))
                
        return issues
        
    def _detect_functional_issues(self, driver: webdriver.Chrome, 
                                page_state: PageState) -> List[DiscoveredIssue]:
        """Detect functional issues."""
        issues = []
        
        # Check for empty required form fields without validation
        for form in page_state.forms:
            for inp in form.get('inputs', []):
                if inp.get('required') and inp.get('type') not in ['submit', 'button', 'hidden']:
                    # This would need more sophisticated checking
                    # For now, just report that required fields exist
                    pass
                    
        return issues


class AIExploratoryTester:
    """Main AI-powered exploratory testing system."""
    
    def __init__(self, base_url: str, max_depth: int = 3, max_pages: int = 50):
        self.base_url = base_url.rstrip('/')
        self.max_depth = max_depth
        self.max_pages = max_pages
        
        # Components
        self.page_analyzer = WebPageAnalyzer()
        self.input_generator = IntelligentInputGenerator()
        self.issue_detector = IssueDetector()
        
        # State tracking
        self.visited_pages = set()
        self.page_states = {}
        self.exploration_graph = nx.DiGraph()
        self.discovered_issues = []
        self.exploration_log = []
        
        # Configuration
        self.strategy = ExplorationStrategy.SMART
        self.interaction_timeout = 10
        self.page_load_timeout = 30
        
    def setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome WebDriver."""
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        # Enable logging
        options.add_argument("--enable-logging")
        options.add_argument("--log-level=0")
        
        # Performance settings
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(self.page_load_timeout)
        driver.implicitly_wait(5)
        
        return driver
        
    def explore_application(self) -> Dict[str, Any]:
        """Start exploratory testing of the application."""
        driver = self.setup_driver()
        
        try:
            logging.info(f"Starting exploration of {self.base_url}")
            
            # Start with the base URL
            self._explore_page(driver, self.base_url, depth=0)
            
            # Continue exploration based on strategy
            self._continue_exploration(driver)
            
            # Final analysis
            results = self._generate_exploration_report()
            
            logging.info(f"Exploration completed. Visited {len(self.visited_pages)} pages, "
                        f"found {len(self.discovered_issues)} issues")
            
            return results
            
        finally:
            driver.quit()
            
    def _explore_page(self, driver: webdriver.Chrome, url: str, depth: int) -> PageState:
        """Explore a single page."""
        if (url in self.visited_pages or 
            depth > self.max_depth or 
            len(self.visited_pages) >= self.max_pages):
            return None
            
        try:
            logging.info(f"Exploring page: {url} (depth: {depth})")
            
            # Navigate to page
            driver.get(url)
            time.sleep(2)  # Allow page to load
            
            # Analyze page state
            page_state = self.page_analyzer.analyze_page(driver)
            
            # Mark as visited
            self.visited_pages.add(url)
            self.page_states[url] = page_state
            
            # Add to exploration graph
            self.exploration_graph.add_node(url, **{
                'title': page_state.title,
                'depth': depth,
                'interactive_elements': len(page_state.interactive_elements),
                'forms': len(page_state.forms),
                'links': len(page_state.links)
            })
            
            # Detect issues
            issues = self.issue_detector.detect_issues(driver, page_state)
            self.discovered_issues.extend(issues)
            
            # Interact with page elements
            self._interact_with_page(driver, page_state, depth)
            
            return page_state
            
        except Exception as e:
            logging.error(f"Error exploring page {url}: {e}")
            return None
            
    def _interact_with_page(self, driver: webdriver.Chrome, 
                           page_state: PageState, depth: int):
        """Interact with elements on the current page."""
        # Interact with forms
        self._interact_with_forms(driver, page_state.forms)
        
        # Click interactive elements
        self._interact_with_elements(driver, page_state.interactive_elements)
        
        # Follow links (if not at max depth)
        if depth < self.max_depth:
            self._follow_links(driver, page_state.links, depth + 1)
            
    def _interact_with_forms(self, driver: webdriver.Chrome, forms: List[Dict[str, Any]]):
        """Interact with forms on the page."""
        for form in forms:
            try:
                form_element = driver.find_element(By.XPATH, form['xpath'])
                
                # Fill form inputs
                for input_info in form['inputs']:
                    if input_info['type'] in ['submit', 'button', 'hidden']:
                        continue
                        
                    try:
                        input_element = driver.find_element(By.XPATH, input_info['xpath'])
                        
                        if input_info['type'] == 'select':
                            # Handle select elements
                            select = Select(input_element)
                            if select.options:
                                select.select_by_index(1)  # Select second option
                        else:
                            # Generate and input text
                            test_input = self.input_generator.generate_input(input_info)
                            input_element.clear()
                            input_element.send_keys(test_input)
                            
                        # Log the interaction
                        action = ExplorationAction(
                            action_type=InteractionType.INPUT,
                            element_info=input_info,
                            input_data=test_input if input_info['type'] != 'select' else 'selected_option',
                            result='success',
                            before_state=driver.current_url,
                            after_state=driver.current_url
                        )
                        self.exploration_log.append(action)
                        
                    except Exception as e:
                        logging.debug(f"Error filling input: {e}")
                        continue
                        
                # Optionally submit form (be careful with this)
                # self._submit_form_safely(driver, form_element)
                
            except Exception as e:
                logging.debug(f"Error interacting with form: {e}")
                
    def _interact_with_elements(self, driver: webdriver.Chrome, 
                              elements: List[Dict[str, Any]]):
        """Interact with clickable elements."""
        # Limit interactions to avoid too many actions
        max_interactions = min(10, len(elements))
        selected_elements = random.sample(elements, max_interactions) if len(elements) > max_interactions else elements
        
        for element_info in selected_elements:
            if element_info['tag'] in ['button', 'a'] and element_info['text']:
                try:
                    # Find element by xpath or other means
                    xpath = element_info.get('xpath')
                    if not xpath:
                        continue
                        
                    element = driver.find_element(By.XPATH, xpath)
                    
                    # Skip if element is not interactable
                    if not (element.is_displayed() and element.is_enabled()):
                        continue
                        
                    before_url = driver.current_url
                    
                    # Click element
                    element.click()
                    time.sleep(1)  # Wait for response
                    
                    after_url = driver.current_url
                    
                    # Log the interaction
                    action = ExplorationAction(
                        action_type=InteractionType.CLICK,
                        element_info=element_info,
                        input_data=None,
                        result='success' if before_url != after_url else 'no_change',
                        before_state=before_url,
                        after_state=after_url
                    )
                    self.exploration_log.append(action)
                    
                    # If URL changed, we might have navigated to a new page
                    if before_url != after_url:
                        # Add edge to graph
                        self.exploration_graph.add_edge(before_url, after_url)
                        
                except Exception as e:
                    logging.debug(f"Error clicking element: {e}")
                    action = ExplorationAction(
                        action_type=InteractionType.CLICK,
                        element_info=element_info,
                        input_data=None,
                        result='error',
                        before_state=driver.current_url,
                        after_state=driver.current_url
                    )
                    self.exploration_log.append(action)
                    
    def _follow_links(self, driver: webdriver.Chrome, links: List[Dict[str, Any]], depth: int):
        """Follow links to explore new pages."""
        # Filter internal links
        internal_links = []
        base_domain = urlparse(self.base_url).netloc
        
        for link in links:
            href = link['href']
            
            # Make absolute URL
            if href.startswith('/'):
                full_url = urljoin(self.base_url, href)
            elif href.startswith('http'):
                full_url = href
            else:
                continue
                
            # Check if it's an internal link
            link_domain = urlparse(full_url).netloc
            if link_domain == base_domain or link_domain == '':
                internal_links.append(full_url)
                
        # Randomly select links to follow (avoid following all links)
        max_links = min(5, len(internal_links))
        selected_links = random.sample(internal_links, max_links) if len(internal_links) > max_links else internal_links
        
        for link_url in selected_links:
            if link_url not in self.visited_pages:
                # Add edge to graph
                current_url = driver.current_url
                self.exploration_graph.add_edge(current_url, link_url)
                
                # Explore the linked page
                self._explore_page(driver, link_url, depth)
                
                # Navigate back to continue exploration
                try:
                    driver.get(current_url)
                    time.sleep(1)
                except:
                    pass
                    
    def _continue_exploration(self, driver: webdriver.Chrome):
        """Continue exploration based on strategy."""
        if self.strategy == ExplorationStrategy.BREADTH_FIRST:
            self._breadth_first_exploration(driver)
        elif self.strategy == ExplorationStrategy.DEPTH_FIRST:
            self._depth_first_exploration(driver)
        elif self.strategy == ExplorationStrategy.RANDOM:
            self._random_exploration(driver)
        elif self.strategy == ExplorationStrategy.SMART:
            self._smart_exploration(driver)
        elif self.strategy == ExplorationStrategy.MODEL_BASED:
            self._model_based_exploration(driver)
            
    def _breadth_first_exploration(self, driver: webdriver.Chrome):
        """Breadth-first exploration strategy."""
        queue = deque([(self.base_url, 0)])
        
        while queue and len(self.visited_pages) < self.max_pages:
            url, depth = queue.popleft()
            
            if url not in self.visited_pages and depth <= self.max_depth:
                page_state = self._explore_page(driver, url, depth)
                
                if page_state:
                    # Add linked pages to queue
                    for link in page_state.links:
                        queue.append((link['href'], depth + 1))
                        
    def _smart_exploration(self, driver: webdriver.Chrome):
        """Smart exploration that prioritizes interesting pages."""
        # Already implemented in the main explore_page method
        # This could be enhanced with ML-based prioritization
        pass
        
    def _model_based_exploration(self, driver: webdriver.Chrome):
        """Model-based exploration using application model."""
        # This would implement a more sophisticated model-based approach
        # For now, use smart exploration
        self._smart_exploration(driver)
        
    def _random_exploration(self, driver: webdriver.Chrome):
        """Random exploration strategy."""
        unvisited_urls = []
        
        # Collect all discovered URLs
        for page_state in self.page_states.values():
            for link in page_state.links:
                if link['href'] not in self.visited_pages:
                    unvisited_urls.append(link['href'])
                    
        # Randomly explore unvisited URLs
        random.shuffle(unvisited_urls)
        
        for url in unvisited_urls[:self.max_pages - len(self.visited_pages)]:
            self._explore_page(driver, url, 1)  # Depth 1 for random exploration
            
    def _depth_first_exploration(self, driver: webdriver.Chrome):
        """Depth-first exploration strategy."""
        def dfs(url, depth):
            if url in self.visited_pages or depth > self.max_depth or len(self.visited_pages) >= self.max_pages:
                return
                
            page_state = self._explore_page(driver, url, depth)
            
            if page_state:
                for link in page_state.links:
                    dfs(link['href'], depth + 1)
                    
        dfs(self.base_url, 0)
        
    def _generate_exploration_report(self) -> Dict[str, Any]:
        """Generate comprehensive exploration report."""
        # Calculate statistics
        total_pages = len(self.visited_pages)
        total_issues = len(self.discovered_issues)
        
        # Issue statistics
        issue_by_type = defaultdict(int)
        issue_by_severity = defaultdict(int)
        
        for issue in self.discovered_issues:
            issue_by_type[issue.issue_type.value] += 1
            issue_by_severity[issue.severity] += 1
            
        # Page statistics
        page_stats = {
            'total_pages': total_pages,
            'avg_load_time': np.mean([
                page.performance_metrics.get('load_time', 0) 
                for page in self.page_states.values()
            ]) if self.page_states else 0,
            'pages_with_forms': len([p for p in self.page_states.values() if p.forms]),
            'pages_with_errors': len([p for p in self.page_states.values() if p.errors])
        }
        
        # Coverage analysis
        coverage_stats = {
            'unique_pages_found': total_pages,
            'total_interactions': len(self.exploration_log),
            'successful_interactions': len([a for a in self.exploration_log if a.result == 'success']),
            'forms_tested': len(set(a.element_info.get('xpath', '') for a in self.exploration_log 
                                  if a.action_type == InteractionType.INPUT))
        }
        
        return {
            'summary': {
                'base_url': self.base_url,
                'exploration_strategy': self.strategy.value,
                'total_pages_visited': total_pages,
                'total_issues_found': total_issues,
                'exploration_depth': self.max_depth,
                'start_time': datetime.now().isoformat()
            },
            'page_statistics': page_stats,
            'coverage_statistics': coverage_stats,
            'issues': {
                'total': total_issues,
                'by_type': dict(issue_by_type),
                'by_severity': dict(issue_by_severity),
                'details': [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description,
                        'location': issue.location,
                        'confidence': issue.confidence,
                        'evidence': issue.evidence
                    }
                    for issue in self.discovered_issues
                ]
            },
            'exploration_graph': {
                'nodes': len(self.exploration_graph.nodes),
                'edges': len(self.exploration_graph.edges),
                'pages': list(self.visited_pages)
            },
            'interactions': [
                {
                    'type': action.action_type.value,
                    'result': action.result,
                    'before_state': action.before_state,
                    'after_state': action.after_state,
                    'timestamp': action.timestamp.isoformat()
                }
                for action in self.exploration_log
            ]
        }
        
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save exploration report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logging.info(f"Exploration report saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    tester = AIExploratoryTester(
        base_url="http://localhost:3000",
        max_depth=2,
        max_pages=20
    )
    
    # Set exploration strategy
    tester.strategy = ExplorationStrategy.SMART
    
    # Run exploration
    report = tester.explore_application()
    
    # Save report
    tester.save_report(report, "exploratory_test_report.json")
    
    # Print summary
    print(f"Exploration completed!")
    print(f"Pages visited: {report['summary']['total_pages_visited']}")
    print(f"Issues found: {report['summary']['total_issues_found']}")
    print(f"Success rate: {report['coverage_statistics']['successful_interactions']/max(report['coverage_statistics']['total_interactions'], 1):.1%}")
    
    # Print top issues
    if report['issues']['details']:
        print("\nTop Issues:")
        for issue in sorted(report['issues']['details'], 
                           key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x['severity'], 0), 
                           reverse=True)[:5]:
            print(f"  - {issue['type']}: {issue['description']} ({issue['severity']})")