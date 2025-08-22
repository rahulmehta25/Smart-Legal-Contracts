"""
AI-Powered Visual Regression Testing

This module provides intelligent visual regression testing using computer vision
and machine learning to detect visual changes and anomalies in web applications.
"""

import base64
import cv2
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests


class ChangeType(Enum):
    LAYOUT = "layout"
    COLOR = "color"
    TEXT = "text"
    CONTENT = "content"
    POSITIONING = "positioning"
    STYLING = "styling"
    ANIMATION = "animation"
    RESPONSIVE = "responsive"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    COSMETIC = "cosmetic"


@dataclass
class VisualDifference:
    """Represents a detected visual difference."""
    change_type: ChangeType
    severity: Severity
    confidence: float
    region: Tuple[int, int, int, int]  # x, y, width, height
    description: str
    before_image_path: str
    after_image_path: str
    diff_image_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenshotMetadata:
    """Metadata for a screenshot."""
    url: str
    viewport_size: Tuple[int, int]
    device_type: str
    timestamp: datetime
    browser: str
    user_agent: str
    page_load_time: float
    dom_elements_count: int
    css_selectors: List[str] = field(default_factory=list)


@dataclass
class VisualTestResult:
    """Result of a visual test."""
    test_name: str
    baseline_path: str
    current_path: str
    differences: List[VisualDifference]
    overall_similarity: float
    test_passed: bool
    execution_time: float
    metadata: ScreenshotMetadata


class ImagePreprocessor:
    """Preprocesses images for comparison."""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_for_ml(self, image_path: str) -> torch.Tensor:
        """Preprocess image for ML model."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract visual features from image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract features
        features = []
        
        # Color histogram features
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        features.extend([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        
        # Texture features using LBP (Local Binary Patterns)
        lbp = self._calculate_lbp(gray)
        features.append(cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten())
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.append(cv2.calcHist([edges], [0], None, [256], [0, 256]).flatten())
        
        # Structural features
        structural_features = self._extract_structural_features(gray)
        features.append(structural_features)
        
        return np.concatenate(features)
        
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        # Simplified LBP implementation
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                
                # 8-connected neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                # Calculate LBP value
                lbp_value = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        lbp_value += 2**k
                        
                lbp[i, j] = lbp_value
                
        return lbp
        
    def _extract_structural_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract structural features from image."""
        # Calculate moments
        moments = cv2.moments(gray_image)
        
        # Hu moments (translation, scale, rotation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Statistical features
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        skewness = self._calculate_skewness(gray_image)
        kurtosis = self._calculate_kurtosis(gray_image)
        
        # Combine features
        features = np.concatenate([
            hu_moments,
            [mean_intensity, std_intensity, skewness, kurtosis]
        ])
        
        return features
        
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensities."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image intensities."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 4) - 3


class VisualDifferenceDetector:
    """Detects visual differences between images using multiple algorithms."""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        
    def detect_differences(self, baseline_path: str, current_path: str,
                          threshold: float = 0.1) -> List[VisualDifference]:
        """Detect all types of visual differences."""
        differences = []
        
        # Load images
        baseline = cv2.imread(baseline_path)
        current = cv2.imread(current_path)
        
        if baseline is None or current is None:
            raise ValueError("Could not load images for comparison")
            
        # Ensure images have same dimensions
        if baseline.shape != current.shape:
            current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))
            
        # Detect different types of changes
        layout_diffs = self._detect_layout_changes(baseline, current, threshold)
        color_diffs = self._detect_color_changes(baseline, current, threshold)
        text_diffs = self._detect_text_changes(baseline_path, current_path, threshold)
        content_diffs = self._detect_content_changes(baseline, current, threshold)
        
        differences.extend(layout_diffs)
        differences.extend(color_diffs)
        differences.extend(text_diffs)
        differences.extend(content_diffs)
        
        return differences
        
    def _detect_layout_changes(self, baseline: np.ndarray, current: np.ndarray,
                              threshold: float) -> List[VisualDifference]:
        """Detect layout and positioning changes."""
        differences = []
        
        # Convert to grayscale
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        baseline_edges = cv2.Canny(baseline_gray, 50, 150)
        current_edges = cv2.Canny(current_gray, 50, 150)
        
        # Calculate difference
        edge_diff = cv2.absdiff(baseline_edges, current_edges)
        
        # Find contours of differences
        contours, _ = cv2.findContours(edge_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on change area
                total_pixels = baseline.shape[0] * baseline.shape[1]
                confidence = min(area / total_pixels * 10, 1.0)
                
                if confidence > threshold:
                    severity = self._determine_severity(confidence, area)
                    
                    differences.append(VisualDifference(
                        change_type=ChangeType.LAYOUT,
                        severity=severity,
                        confidence=confidence,
                        region=(x, y, w, h),
                        description=f"Layout change detected in region ({x}, {y}, {w}, {h})",
                        before_image_path="",
                        after_image_path="",
                        diff_image_path="",
                        metadata={"area": area, "contour_points": len(contour)}
                    ))
                    
        return differences
        
    def _detect_color_changes(self, baseline: np.ndarray, current: np.ndarray,
                             threshold: float) -> List[VisualDifference]:
        """Detect color changes."""
        differences = []
        
        # Convert to different color spaces for better detection
        baseline_hsv = cv2.cvtColor(baseline, cv2.COLOR_BGR2HSV)
        current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
        
        # Calculate differences in each channel
        h_diff = cv2.absdiff(baseline_hsv[:,:,0], current_hsv[:,:,0])
        s_diff = cv2.absdiff(baseline_hsv[:,:,1], current_hsv[:,:,1])
        v_diff = cv2.absdiff(baseline_hsv[:,:,2], current_hsv[:,:,2])
        
        # Combine differences
        combined_diff = cv2.bitwise_or(cv2.bitwise_or(h_diff, s_diff), v_diff)
        
        # Threshold the differences
        _, thresholded = cv2.threshold(combined_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate average color difference in region
                region_baseline = baseline_hsv[y:y+h, x:x+w]
                region_current = current_hsv[y:y+h, x:x+w]
                
                avg_diff = np.mean(cv2.absdiff(region_baseline, region_current))
                confidence = min(avg_diff / 255.0 * 3, 1.0)
                
                if confidence > threshold:
                    severity = self._determine_severity(confidence, area)
                    
                    differences.append(VisualDifference(
                        change_type=ChangeType.COLOR,
                        severity=severity,
                        confidence=confidence,
                        region=(x, y, w, h),
                        description=f"Color change detected (avg diff: {avg_diff:.1f})",
                        before_image_path="",
                        after_image_path="",
                        diff_image_path="",
                        metadata={"average_difference": avg_diff, "area": area}
                    ))
                    
        return differences
        
    def _detect_text_changes(self, baseline_path: str, current_path: str,
                           threshold: float) -> List[VisualDifference]:
        """Detect text changes using OCR."""
        differences = []
        
        try:
            import pytesseract
            
            # Extract text from both images
            baseline_text = pytesseract.image_to_string(Image.open(baseline_path))
            current_text = pytesseract.image_to_string(Image.open(current_path))
            
            # Compare text content
            if baseline_text.strip() != current_text.strip():
                # Calculate text similarity
                baseline_words = set(baseline_text.lower().split())
                current_words = set(current_text.lower().split())
                
                if baseline_words or current_words:
                    similarity = len(baseline_words & current_words) / len(baseline_words | current_words)
                    confidence = 1.0 - similarity
                    
                    if confidence > threshold:
                        differences.append(VisualDifference(
                            change_type=ChangeType.TEXT,
                            severity=Severity.MEDIUM,
                            confidence=confidence,
                            region=(0, 0, 0, 0),  # Full image
                            description="Text content has changed",
                            before_image_path=baseline_path,
                            after_image_path=current_path,
                            diff_image_path="",
                            metadata={
                                "baseline_text": baseline_text[:200],
                                "current_text": current_text[:200],
                                "similarity": similarity
                            }
                        ))
                        
        except ImportError:
            logging.warning("pytesseract not available, skipping text detection")
        except Exception as e:
            logging.error(f"Error in text detection: {e}")
            
        return differences
        
    def _detect_content_changes(self, baseline: np.ndarray, current: np.ndarray,
                              threshold: float) -> List[VisualDifference]:
        """Detect general content changes."""
        differences = []
        
        # Calculate structural similarity
        ssim_score = self._calculate_ssim(baseline, current)
        
        if ssim_score < (1.0 - threshold):
            confidence = 1.0 - ssim_score
            
            # Use template matching to find significant changes
            diff = cv2.absdiff(baseline, current)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find significant changes
            _, thresholded = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = np.ones((5,5), np.uint8)
            cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            
            # Find largest change region
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                area = cv2.contourArea(largest_contour)
                
                if area > 200:  # Minimum significant change
                    severity = self._determine_severity(confidence, area)
                    
                    differences.append(VisualDifference(
                        change_type=ChangeType.CONTENT,
                        severity=severity,
                        confidence=confidence,
                        region=(x, y, w, h),
                        description=f"Content change detected (SSIM: {ssim_score:.3f})",
                        before_image_path="",
                        after_image_path="",
                        diff_image_path="",
                        metadata={"ssim_score": ssim_score, "change_area": area}
                    ))
                    
        return differences
        
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate means
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # Calculate variances and covariance
        var1 = np.var(gray1)
        var2 = np.var(gray2)
        cov = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
               
        return max(0, min(ssim, 1))
        
    def _determine_severity(self, confidence: float, area: float) -> Severity:
        """Determine severity based on confidence and area."""
        if confidence > 0.8 and area > 10000:
            return Severity.CRITICAL
        elif confidence > 0.6 and area > 5000:
            return Severity.HIGH
        elif confidence > 0.4 and area > 1000:
            return Severity.MEDIUM
        elif confidence > 0.2:
            return Severity.LOW
        else:
            return Severity.COSMETIC


class ScreenshotCapture:
    """Captures screenshots with various configurations."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.drivers = {}
        
    def setup_driver(self, browser: str = "chrome", device_type: str = "desktop") -> webdriver:
        """Setup WebDriver with specified configuration."""
        driver_key = f"{browser}_{device_type}"
        
        if driver_key not in self.drivers:
            if browser.lower() == "chrome":
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                
                # Device-specific settings
                if device_type == "mobile":
                    options.add_argument("--window-size=375,667")  # iPhone 6/7/8
                    mobile_emulation = {"deviceName": "iPhone 6"}
                    options.add_experimental_option("mobileEmulation", mobile_emulation)
                elif device_type == "tablet":
                    options.add_argument("--window-size=768,1024")  # iPad
                else:  # desktop
                    options.add_argument("--window-size=1920,1080")
                    
                driver = webdriver.Chrome(options=options)
                
            elif browser.lower() == "firefox":
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument("--headless")
                    
                if device_type == "mobile":
                    options.add_argument("--width=375")
                    options.add_argument("--height=667")
                elif device_type == "tablet":
                    options.add_argument("--width=768")
                    options.add_argument("--height=1024")
                else:
                    options.add_argument("--width=1920")
                    options.add_argument("--height=1080")
                    
                driver = webdriver.Firefox(options=options)
            else:
                raise ValueError(f"Unsupported browser: {browser}")
                
            self.drivers[driver_key] = driver
            
        return self.drivers[driver_key]
        
    def capture_screenshot(self, url: str, output_path: str,
                          browser: str = "chrome", device_type: str = "desktop",
                          wait_for_element: Optional[str] = None,
                          wait_time: float = 2.0) -> ScreenshotMetadata:
        """Capture screenshot with metadata."""
        start_time = time.time()
        
        driver = self.setup_driver(browser, device_type)
        
        try:
            # Navigate to URL
            driver.get(url)
            
            # Wait for specific element if provided
            if wait_for_element:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element)))
            else:
                time.sleep(wait_time)  # Generic wait
                
            # Get page information
            viewport_size = driver.execute_script("return [window.innerWidth, window.innerHeight];")
            dom_elements = len(driver.find_elements(By.XPATH, "//*"))
            user_agent = driver.execute_script("return navigator.userAgent;")
            
            # Capture screenshot
            driver.save_screenshot(output_path)
            
            page_load_time = time.time() - start_time
            
            # Extract CSS selectors (sample)
            css_selectors = []
            try:
                elements = driver.find_elements(By.XPATH, "//*[@class or @id]")[:20]  # Limit to 20
                for element in elements:
                    if element.get_attribute("id"):
                        css_selectors.append(f"#{element.get_attribute('id')}")
                    elif element.get_attribute("class"):
                        classes = element.get_attribute("class").split()
                        if classes:
                            css_selectors.append(f".{classes[0]}")
            except:
                pass
                
            return ScreenshotMetadata(
                url=url,
                viewport_size=tuple(viewport_size),
                device_type=device_type,
                timestamp=datetime.now(),
                browser=browser,
                user_agent=user_agent,
                page_load_time=page_load_time,
                dom_elements_count=dom_elements,
                css_selectors=css_selectors
            )
            
        except Exception as e:
            logging.error(f"Error capturing screenshot for {url}: {e}")
            raise
            
    def capture_full_page_screenshot(self, url: str, output_path: str,
                                   browser: str = "chrome") -> ScreenshotMetadata:
        """Capture full page screenshot (including scrollable content)."""
        driver = self.setup_driver(browser)
        
        try:
            driver.get(url)
            time.sleep(2)
            
            # Get page dimensions
            total_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            
            # Take multiple screenshots and stitch them
            screenshots = []
            scroll_position = 0
            
            while scroll_position < total_height:
                driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(0.5)  # Wait for scroll
                
                screenshot_path = f"/tmp/screenshot_{scroll_position}.png"
                driver.save_screenshot(screenshot_path)
                screenshots.append(screenshot_path)
                
                scroll_position += viewport_height
                
            # Stitch screenshots together
            self._stitch_screenshots(screenshots, output_path)
            
            # Clean up temporary files
            for screenshot in screenshots:
                try:
                    os.remove(screenshot)
                except:
                    pass
                    
            return ScreenshotMetadata(
                url=url,
                viewport_size=(driver.execute_script("return window.innerWidth"), total_height),
                device_type="desktop",
                timestamp=datetime.now(),
                browser=browser,
                user_agent=driver.execute_script("return navigator.userAgent;"),
                page_load_time=0.0,  # Not measured for full page
                dom_elements_count=len(driver.find_elements(By.XPATH, "//*"))
            )
            
        except Exception as e:
            logging.error(f"Error capturing full page screenshot: {e}")
            raise
            
    def _stitch_screenshots(self, screenshot_paths: List[str], output_path: str):
        """Stitch multiple screenshots into one image."""
        if not screenshot_paths:
            return
            
        # Load first image to get dimensions
        first_img = cv2.imread(screenshot_paths[0])
        height, width = first_img.shape[:2]
        
        # Calculate total height
        total_height = height * len(screenshot_paths)
        
        # Create blank canvas
        stitched = np.zeros((total_height, width, 3), dtype=np.uint8)
        
        # Add each screenshot to canvas
        current_y = 0
        for screenshot_path in screenshot_paths:
            img = cv2.imread(screenshot_path)
            if img is not None:
                img_height = img.shape[0]
                stitched[current_y:current_y + img_height, :] = img
                current_y += img_height
                
        # Save stitched image
        cv2.imwrite(output_path, stitched)
        
    def cleanup(self):
        """Clean up WebDriver instances."""
        for driver in self.drivers.values():
            try:
                driver.quit()
            except:
                pass
        self.drivers.clear()


class AIVisualTester:
    """Main AI-powered visual regression testing system."""
    
    def __init__(self, baseline_dir: str, results_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.results_dir = Path(results_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.screenshot_capture = ScreenshotCapture()
        self.difference_detector = VisualDifferenceDetector()
        
        # ML models for advanced analysis
        self.anomaly_threshold = 0.15
        self.similarity_threshold = 0.8
        
    def create_baseline(self, test_config: Dict[str, Any]) -> str:
        """Create baseline screenshots for a test configuration."""
        test_name = test_config.get('name', 'unnamed_test')
        url = test_config['url']
        
        baseline_path = self.baseline_dir / f"{test_name}_baseline.png"
        
        # Capture baseline screenshot
        metadata = self.screenshot_capture.capture_screenshot(
            url=url,
            output_path=str(baseline_path),
            browser=test_config.get('browser', 'chrome'),
            device_type=test_config.get('device', 'desktop'),
            wait_for_element=test_config.get('wait_for_element'),
            wait_time=test_config.get('wait_time', 2.0)
        )
        
        # Save metadata
        metadata_path = self.baseline_dir / f"{test_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'url': metadata.url,
                'viewport_size': metadata.viewport_size,
                'device_type': metadata.device_type,
                'timestamp': metadata.timestamp.isoformat(),
                'browser': metadata.browser,
                'user_agent': metadata.user_agent,
                'page_load_time': metadata.page_load_time,
                'dom_elements_count': metadata.dom_elements_count,
                'css_selectors': metadata.css_selectors
            }, indent=2)
            
        logging.info(f"Baseline created: {baseline_path}")
        return str(baseline_path)
        
    def run_visual_test(self, test_config: Dict[str, Any]) -> VisualTestResult:
        """Run visual regression test."""
        start_time = time.time()
        
        test_name = test_config.get('name', 'unnamed_test')
        url = test_config['url']
        
        # Paths
        baseline_path = self.baseline_dir / f"{test_name}_baseline.png"
        current_path = self.results_dir / f"{test_name}_current.png"
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline not found: {baseline_path}")
            
        # Capture current screenshot
        metadata = self.screenshot_capture.capture_screenshot(
            url=url,
            output_path=str(current_path),
            browser=test_config.get('browser', 'chrome'),
            device_type=test_config.get('device', 'desktop'),
            wait_for_element=test_config.get('wait_for_element'),
            wait_time=test_config.get('wait_time', 2.0)
        )
        
        # Detect differences
        differences = self.difference_detector.detect_differences(
            str(baseline_path),
            str(current_path),
            threshold=test_config.get('threshold', 0.1)
        )
        
        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(
            str(baseline_path), str(current_path)
        )
        
        # Generate difference visualization
        if differences:
            diff_path = self.results_dir / f"{test_name}_diff.png"
            self._create_difference_image(
                str(baseline_path), str(current_path), str(diff_path), differences
            )
            
            # Update difference paths
            for diff in differences:
                diff.before_image_path = str(baseline_path)
                diff.after_image_path = str(current_path)
                diff.diff_image_path = str(diff_path)
        
        # Determine if test passed
        test_passed = (
            overall_similarity >= self.similarity_threshold and
            not any(d.severity in [Severity.CRITICAL, Severity.HIGH] for d in differences)
        )
        
        execution_time = time.time() - start_time
        
        result = VisualTestResult(
            test_name=test_name,
            baseline_path=str(baseline_path),
            current_path=str(current_path),
            differences=differences,
            overall_similarity=overall_similarity,
            test_passed=test_passed,
            execution_time=execution_time,
            metadata=metadata
        )
        
        # Save test result
        self._save_test_result(result)
        
        return result
        
    def _calculate_overall_similarity(self, baseline_path: str, current_path: str) -> float:
        """Calculate overall similarity between two images."""
        try:
            # Load images
            baseline = cv2.imread(baseline_path)
            current = cv2.imread(current_path)
            
            if baseline is None or current is None:
                return 0.0
                
            # Ensure same dimensions
            if baseline.shape != current.shape:
                current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))
                
            # Calculate multiple similarity metrics
            ssim = self.difference_detector._calculate_ssim(baseline, current)
            
            # Histogram correlation
            hist_baseline = cv2.calcHist([baseline], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            hist_current = cv2.calcHist([current], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            hist_corr = cv2.compareHist(hist_baseline, hist_current, cv2.HISTCMP_CORREL)
            
            # Template matching
            result = cv2.matchTemplate(baseline, current, cv2.TM_CCOEFF_NORMED)
            template_match = np.max(result)
            
            # Combined similarity score
            similarity = (ssim * 0.5 + hist_corr * 0.3 + template_match * 0.2)
            return max(0.0, min(similarity, 1.0))
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def _create_difference_image(self, baseline_path: str, current_path: str,
                               diff_path: str, differences: List[VisualDifference]):
        """Create a visual representation of differences."""
        # Load baseline image
        baseline = cv2.imread(baseline_path)
        
        if baseline is None:
            return
            
        # Create difference overlay
        overlay = baseline.copy()
        
        # Draw bounding boxes for each difference
        colors = {
            Severity.CRITICAL: (0, 0, 255),      # Red
            Severity.HIGH: (0, 128, 255),        # Orange
            Severity.MEDIUM: (0, 255, 255),      # Yellow
            Severity.LOW: (0, 255, 0),           # Green
            Severity.COSMETIC: (255, 0, 255)     # Magenta
        }
        
        for diff in differences:
            x, y, w, h = diff.region
            color = colors.get(diff.severity, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Add severity label
            label = f"{diff.severity.value} ({diff.confidence:.2f})"
            cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # Save difference image
        cv2.imwrite(diff_path, overlay)
        
    def _save_test_result(self, result: VisualTestResult):
        """Save test result to JSON file."""
        result_data = {
            'test_name': result.test_name,
            'baseline_path': result.baseline_path,
            'current_path': result.current_path,
            'overall_similarity': result.overall_similarity,
            'test_passed': result.test_passed,
            'execution_time': result.execution_time,
            'timestamp': datetime.now().isoformat(),
            'differences': [
                {
                    'change_type': diff.change_type.value,
                    'severity': diff.severity.value,
                    'confidence': diff.confidence,
                    'region': diff.region,
                    'description': diff.description,
                    'metadata': diff.metadata
                }
                for diff in result.differences
            ],
            'metadata': {
                'url': result.metadata.url,
                'viewport_size': result.metadata.viewport_size,
                'device_type': result.metadata.device_type,
                'browser': result.metadata.browser,
                'page_load_time': result.metadata.page_load_time,
                'dom_elements_count': result.metadata.dom_elements_count
            }
        }
        
        result_file = self.results_dir / f"{result.test_name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
            
    def run_test_suite(self, test_configs: List[Dict[str, Any]]) -> List[VisualTestResult]:
        """Run multiple visual tests."""
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_config = {
                executor.submit(self.run_visual_test, config): config 
                for config in test_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Test completed: {result.test_name} - {'PASS' if result.test_passed else 'FAIL'}")
                except Exception as e:
                    logging.error(f"Test failed for {config.get('name', 'unknown')}: {e}")
                    
        return results
        
    def generate_html_report(self, results: List[VisualTestResult], output_path: str):
        """Generate HTML report of visual test results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visual Regression Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .test-result { border: 1px solid #ddd; margin: 20px 0; padding: 15px; }
                .pass { border-left: 5px solid green; }
                .fail { border-left: 5px solid red; }
                .image-comparison { display: flex; gap: 10px; }
                .image-container { text-align: center; }
                .image-container img { max-width: 300px; border: 1px solid #ddd; }
                .differences { margin: 10px 0; }
                .difference { margin: 5px 0; padding: 5px; background: #f5f5f5; }
                .critical { border-left: 3px solid red; }
                .high { border-left: 3px solid orange; }
                .medium { border-left: 3px solid yellow; }
                .low { border-left: 3px solid green; }
            </style>
        </head>
        <body>
            <h1>Visual Regression Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {total_tests}</p>
                <p>Passed: {passed_tests}</p>
                <p>Failed: {failed_tests}</p>
                <p>Success Rate: {success_rate:.1%}</p>
            </div>
        """.format(
            total_tests=len(results),
            passed_tests=len([r for r in results if r.test_passed]),
            failed_tests=len([r for r in results if not r.test_passed]),
            success_rate=len([r for r in results if r.test_passed]) / len(results) if results else 0
        )
        
        # Add individual test results
        for result in results:
            status_class = "pass" if result.test_passed else "fail"
            
            html_content += f"""
            <div class="test-result {status_class}">
                <h3>{result.test_name} - {'PASS' if result.test_passed else 'FAIL'}</h3>
                <p>Overall Similarity: {result.overall_similarity:.3f}</p>
                <p>Execution Time: {result.execution_time:.2f}s</p>
                <p>URL: {result.metadata.url}</p>
                
                <div class="image-comparison">
                    <div class="image-container">
                        <h4>Baseline</h4>
                        <img src="{os.path.relpath(result.baseline_path, os.path.dirname(output_path))}" alt="Baseline">
                    </div>
                    <div class="image-container">
                        <h4>Current</h4>
                        <img src="{os.path.relpath(result.current_path, os.path.dirname(output_path))}" alt="Current">
                    </div>
            """
            
            # Add diff image if available
            if result.differences:
                diff_path = result.differences[0].diff_image_path
                if diff_path and os.path.exists(diff_path):
                    html_content += f"""
                    <div class="image-container">
                        <h4>Differences</h4>
                        <img src="{os.path.relpath(diff_path, os.path.dirname(output_path))}" alt="Differences">
                    </div>
                    """
            
            html_content += "</div>"
            
            # Add differences list
            if result.differences:
                html_content += "<div class='differences'><h4>Detected Differences:</h4>"
                for diff in result.differences:
                    html_content += f"""
                    <div class="difference {diff.severity.value}">
                        <strong>{diff.change_type.value.title()}</strong> - {diff.severity.value.title()}
                        <br>Confidence: {diff.confidence:.3f}
                        <br>Region: {diff.region}
                        <br>{diff.description}
                    </div>
                    """
                html_content += "</div>"
            
            html_content += "</div>"
            
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logging.info(f"HTML report generated: {output_path}")
        
    def cleanup(self):
        """Clean up resources."""
        self.screenshot_capture.cleanup()


if __name__ == "__main__":
    # Example usage
    tester = AIVisualTester("baselines/", "results/")
    
    # Test configuration
    test_configs = [
        {
            'name': 'homepage_desktop',
            'url': 'http://localhost:3000',
            'browser': 'chrome',
            'device': 'desktop',
            'threshold': 0.1
        },
        {
            'name': 'homepage_mobile',
            'url': 'http://localhost:3000',
            'browser': 'chrome',
            'device': 'mobile',
            'threshold': 0.1
        }
    ]
    
    # Create baselines (first time only)
    for config in test_configs:
        tester.create_baseline(config)
    
    # Run visual tests
    results = tester.run_test_suite(test_configs)
    
    # Generate report
    tester.generate_html_report(results, "visual_test_report.html")
    
    # Cleanup
    tester.cleanup()
    
    # Print results
    for result in results:
        print(f"Test: {result.test_name}")
        print(f"  Status: {'PASS' if result.test_passed else 'FAIL'}")
        print(f"  Similarity: {result.overall_similarity:.3f}")
        print(f"  Differences: {len(result.differences)}")
        for diff in result.differences:
            print(f"    - {diff.change_type.value}: {diff.severity.value} ({diff.confidence:.3f})")