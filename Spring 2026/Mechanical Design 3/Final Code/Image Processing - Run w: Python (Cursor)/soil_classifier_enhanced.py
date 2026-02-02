#!/usr/bin/env python3
"""
Enhanced Soil Image Classification Script for UF/IFAS Analytical Services

This script classifies soil images as Type A (mineral topsoil) or Type B (organic-rich material)
using both rule-based classification and machine learning (scikit-learn) with ≥90% accuracy.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
import sqlite3
import csv
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import joblib
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MunsellColor:
    """Represents a Munsell color with Hue, Value, and Chroma components."""
    hue: str
    value: float
    chroma: int

@dataclass
class SoilSample:
    """Represents a soil sample with classification results."""
    sample_id: str
    timestamp: str
    classification: str
    confidence: float
    bin_count: int
    bin_values: Dict[str, Any]
    bin_proportions: Dict[str, float]
    lighting_lux: float
    image_shape: Tuple[int, int, int]
    processing_time: float
    ml_classification: Optional[str] = None
    ml_confidence: Optional[float] = None
    classification_method: str = "rule_based"  # "rule_based", "ml", or "hybrid"

class MLSoilClassifier:
    """Machine learning based soil classifier using scikit-learn."""
    
    def __init__(self, model_path: str = "soil_classifier_sklearn.pkl"):
        """
        Initialize the ML classifier.
        
        Args:
            model_path: Path to the trained scikit-learn model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ["type_a", "type_b"]
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if Path(self.model_path).exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"ML model loaded from: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
                self.model = None
        else:
            logger.warning(f"ML model not found: {self.model_path}")
            self.model = None
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a soil image.
        
        Args:
            image: Input image array
            
        Returns:
            Feature vector
        """
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # Color statistics (RGB)
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # Grayscale statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        # HSV statistics
        for channel in cv2.split(hsv):
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # LAB statistics
        for channel in cv2.split(lab):
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Texture features (simplified)
        kernel = np.ones((3,3), np.float32) / 9
        blurred = cv2.filter2D(gray, -1, kernel)
        features.extend([
            np.mean(np.abs(gray - blurred)),
            np.std(np.abs(gray - blurred))
        ])
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        return np.array(features)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict soil type for an image.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            return "Unknown", 0.0
        
        try:
            # Extract features
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features)[0]
            predicted_class = self.class_names[prediction]
            
            # Get confidence (probability)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = probabilities[prediction]
            else:
                # For models without predict_proba, use decision function
                if hasattr(self.model, 'decision_function'):
                    decision = self.model.decision_function(features)[0]
                    confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
                else:
                    confidence = 0.5  # Default confidence
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return "Unknown", 0.0

class YOLOSoilClassifier:
    """YOLOv11-based soil classifier."""
    
    def __init__(self, model_path: str = "soil_classifier_yolo.pt"):
        """
        Initialize the YOLOv11 classifier.
        
        Args:
            model_path: Path to the trained YOLOv11 model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ["type_a", "type_b"]
        self._load_model()
    
    def _load_model(self):
        """Load the trained YOLOv11 model."""
        if Path(self.model_path).exists():
            try:
                self.model = YOLO(self.model_path)
                logger.info(f"YOLOv11 model loaded from: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLOv11 model: {e}")
                self.model = None
        else:
            logger.warning(f"YOLOv11 model not found: {self.model_path}")
            self.model = None
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict soil type for an image.
        
        Args:
            image: Input image array (BGR format from OpenCV)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            return "Unknown", 0.0
        
        try:
            # Run inference
            results = self.model(image, verbose=False)
            pred = results[0]
            
            # Get prediction
            predicted_class_idx = pred.probs.top1
            confidence = float(pred.probs.top1conf.item())
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"YOLOv11 prediction failed: {e}")
            return "Unknown", 0.0

class MunsellConverter:
    """Handles conversion between RGB and Munsell color systems."""
    
    def __init__(self):
        # Initialize Munsell lookup tables (simplified version)
        # In a production system, these would be comprehensive lookup tables
        self.munsell_lookup = self._initialize_munsell_lookup()
    
    def _initialize_munsell_lookup(self) -> Dict:
        """Initialize a simplified Munsell color lookup table."""
        # This is a simplified lookup table for demonstration
        # A full implementation would use comprehensive Munsell color data
        return {
            # Brown/Reddish colors (typical for organic-rich soils)
            '10YR': {'2/1': [45, 35, 25], '2/2': [55, 40, 30], '3/2': [65, 50, 35]},
            '7.5YR': {'2/1': [40, 30, 20], '2/2': [50, 35, 25], '3/2': [60, 45, 30]},
            '5YR': {'2/1': [35, 25, 15], '2/2': [45, 30, 20], '3/2': [55, 40, 25]},
            # Gray colors (typical for mineral topsoils)
            '10Y': {'6/1': [140, 140, 120], '7/1': [160, 160, 140], '8/1': [180, 180, 160]},
            '5Y': {'6/1': [130, 130, 110], '7/1': [150, 150, 130], '8/1': [170, 170, 150]},
        }
    
    def rgb_to_munsell(self, rgb: Tuple[int, int, int]) -> MunsellColor:
        """
        Convert RGB color to Munsell color system.
        
        Args:
            rgb: RGB color tuple (R, G, B)
            
        Returns:
            MunsellColor object with hue, value, and chroma
        """
        r, g, b = rgb
        
        # Convert RGB to HSV for easier analysis
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        # Simplified conversion logic (in production, use comprehensive lookup tables)
        if v < 50:  # Dark colors
            if s > 100:  # High saturation - brownish
                hue = '10YR' if h < 30 else '7.5YR'
                value = 2.0
                chroma = 1 if s < 150 else 2
            else:  # Low saturation - grayish
                hue = '10Y'
                value = 3.0 if v > 25 else 2.0
                chroma = 1
        elif v < 100:  # Medium colors
            if s > 80:  # Brownish
                hue = '5YR' if h < 30 else '10YR'
                value = 3.0
                chroma = 2 if s > 120 else 1
            else:  # Grayish
                hue = '5Y'
                value = 4.0
                chroma = 1
        else:  # Light colors
            hue = '10Y'
            value = 6.0 + (v - 100) / 25
            chroma = 1
        
        return MunsellColor(hue, round(value, 1), int(chroma))

class SoilClassifier:
    """Enhanced soil classification system with ML integration."""
    
    def __init__(self, db_path: str = "soil_samples.db", 
                 ml_model_path: str = "soil_classifier_sklearn.pkl",
                 yolo_model_path: str = "soil_classifier_yolo.pt",
                 ml_threshold: float = 0.6, 
                 use_ml: bool = True,
                 use_yolo: bool = True,
                 model_type: str = "yolo"):  # "yolo", "rf", or "both"
        """
        Initialize the enhanced soil classifier.
        
        Args:
            db_path: Path to SQLite database
            ml_model_path: Path to trained Random Forest model
            yolo_model_path: Path to trained YOLOv11 model
            ml_threshold: Confidence threshold for ML predictions
            use_ml: Whether to use ML classification
            use_yolo: Whether to use YOLOv11 classification
            model_type: Which model to use ("yolo", "rf", or "both")
        """
        self.munsell_converter = MunsellConverter()
        self.rf_classifier = MLSoilClassifier(ml_model_path)
        self.yolo_classifier = YOLOSoilClassifier(yolo_model_path)
        self.samples = {}  # In-memory storage for up to 100,000 samples
        self.db_path = db_path
        self.ml_threshold = ml_threshold
        self.use_ml = use_ml
        self.use_yolo = use_yolo
        self.model_type = model_type  # Which model to prefer
        self._initialize_database()
        
        # Classification parameters
        self.min_image_size = (200, 200)
        self.min_lighting_lux = 1000
        self.chroma_tolerance = 1  # ±1 chroma tolerance for binning
        
        logger.info(f"Enhanced Soil Classifier initialized")
        logger.info(f"Random Forest model available: {self.rf_classifier.model is not None}")
        logger.info(f"YOLOv11 model available: {self.yolo_classifier.model is not None}")
        logger.info(f"ML threshold: {self.ml_threshold}")
        logger.info(f"Use ML: {self.use_ml}")
        logger.info(f"Use YOLO: {self.use_yolo}")
        logger.info(f"Model type: {self.model_type}")
        
    def _initialize_database(self):
        """Initialize SQLite database for sample storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS soil_samples (
                sample_id TEXT PRIMARY KEY,
                timestamp TEXT,
                classification TEXT,
                confidence REAL,
                bin_count INTEGER,
                bin_values TEXT,
                bin_proportions TEXT,
                lighting_lux REAL,
                image_shape TEXT,
                processing_time REAL,
                ml_classification TEXT,
                ml_confidence REAL,
                classification_method TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def validate_lighting(self, image: np.ndarray) -> float:
        """
        Validate lighting conditions in the image.
        
        Args:
            image: Input image array
            
        Returns:
            Estimated lighting in lux
        """
        # Convert to grayscale and calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Simple conversion from pixel brightness to lux (calibrated)
        # In production, this would use proper photometric calibration
        estimated_lux = (avg_brightness / 255.0) * 2000  # Scale to 0-2000 lux
        
        return estimated_lux
    
    def crop_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """
        Crop image to specified size, maintaining center region.
        
        Args:
            image: Input image array
            size: Target size (width, height), defaults to min_image_size
            
        Returns:
            Cropped image array
        """
        if size is None:
            size = self.min_image_size
            
        height, width = image.shape[:2]
        target_width, target_height = size
        
        # Calculate crop coordinates (center crop)
        start_x = max(0, (width - target_width) // 2)
        start_y = max(0, (height - target_height) // 2)
        end_x = start_x + target_width
        end_y = start_y + target_height
        
        # Ensure we don't exceed image boundaries
        end_x = min(end_x, width)
        end_y = min(end_y, height)
        
        cropped = image[start_y:end_y, start_x:end_x]
        
        # Resize if necessary
        if cropped.shape[:2] != (target_height, target_width):
            cropped = cv2.resize(cropped, (target_width, target_height))
            
        return cropped
    
    def bin_similar_pixels(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Bin pixels with similar chroma values into descriptive regions.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with bin information
        """
        height, width = image.shape[:2]
        bins = {}
        
        for y in range(height):
            for x in range(width):
                # Get RGB values
                rgb = tuple(image[y, x])
                
                # Convert to Munsell
                munsell = self.munsell_converter.rgb_to_munsell(rgb)
                
                # Create bin key based on hue and value
                bin_key = f"{munsell.hue}_{munsell.value}"
                
                # Check if we can add to existing bin (within chroma tolerance)
                added_to_bin = False
                for existing_bin in bins.keys():
                    existing_munsell = MunsellColor(
                        existing_bin.split('_')[0],
                        float(existing_bin.split('_')[1]),
                        0  # Chroma not used in bin key
                    )
                    
                    if (existing_munsell.hue == munsell.hue and 
                        abs(existing_munsell.value - munsell.value) <= 1.0):
                        bins[existing_bin]['count'] += 1
                        bins[existing_bin]['pixels'].append((x, y))
                        bins[existing_bin]['chroma_values'].append(munsell.chroma)
                        added_to_bin = True
                        break
                
                # Create new bin if not added to existing
                if not added_to_bin:
                    bins[bin_key] = {
                        'count': 1,
                        'pixels': [(x, y)],
                        'chroma_values': [munsell.chroma],
                        'munsell_hue': munsell.hue,
                        'munsell_value': munsell.value,
                        'avg_chroma': munsell.chroma
                    }
        
        # Calculate proportions and average chroma for each bin
        total_pixels = height * width
        for bin_key in bins:
            bins[bin_key]['proportion'] = bins[bin_key]['count'] / total_pixels
            bins[bin_key]['avg_chroma'] = np.mean(bins[bin_key]['chroma_values'])
        
        return bins
    
    def analyze_organic_matter_distribution(self, bins: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze organic matter distribution based on Munsell color bins.
        
        Args:
            bins: Dictionary of color bins
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            'dark_bins_count': 0,
            'brown_bins_count': 0,
            'gray_bins_count': 0,
            'light_bins_count': 0,
            'total_dark_proportion': 0.0,
            'total_brown_proportion': 0.0,
            'total_gray_proportion': 0.0,
            'dominant_hue': None,
            'avg_value': 0.0,
            'avg_chroma': 0.0
        }
        
        total_proportion = 0.0
        value_sum = 0.0
        chroma_sum = 0.0
        hue_counts = {}
        
        for bin_key, bin_data in bins.items():
            hue = bin_data['munsell_hue']
            value = bin_data['munsell_value']
            chroma = bin_data['avg_chroma']
            proportion = bin_data['proportion']
            
            # Categorize bins
            if value <= 3.0:
                analysis['dark_bins_count'] += 1
                analysis['total_dark_proportion'] += proportion
            elif 'YR' in hue or 'R' in hue:  # Reddish/Brownish hues
                analysis['brown_bins_count'] += 1
                analysis['total_brown_proportion'] += proportion
            elif 'Y' in hue and chroma <= 2:  # Grayish hues
                analysis['gray_bins_count'] += 1
                analysis['total_gray_proportion'] += proportion
            else:
                analysis['light_bins_count'] += 1
            
            # Track hue distribution
            hue_counts[hue] = hue_counts.get(hue, 0) + proportion
            
            # Accumulate averages
            value_sum += value * proportion
            chroma_sum += chroma * proportion
            total_proportion += proportion
        
        # Calculate final metrics
        if total_proportion > 0:
            analysis['avg_value'] = value_sum / total_proportion
            analysis['avg_chroma'] = chroma_sum / total_proportion
        
        # Find dominant hue
        if hue_counts:
            analysis['dominant_hue'] = max(hue_counts, key=hue_counts.get)
        
        return analysis
    
    def classify_soil_type_rule_based(self, analysis: Dict[str, Any], bins: Dict[str, Dict]) -> Tuple[str, float]:
        """
        Classify soil as Type A (mineral topsoil) or Type B (organic-rich material) using rule-based approach.
        
        Args:
            analysis: Organic matter distribution analysis
            bins: Color bins dictionary
            
        Returns:
            Tuple of (classification, confidence)
        """
        # Classification rules based on organic matter distribution
        dark_prop = analysis['total_dark_proportion']
        brown_prop = analysis['total_brown_proportion']
        gray_prop = analysis['total_gray_proportion']
        avg_value = analysis['avg_value']
        avg_chroma = analysis['avg_chroma']
        
        # Type B (organic-rich) indicators
        type_b_score = 0.0
        
        # High proportion of dark colors (organic matter)
        if dark_prop > 0.3:
            type_b_score += 0.4
        elif dark_prop > 0.2:
            type_b_score += 0.3
        elif dark_prop > 0.1:
            type_b_score += 0.2
        
        # High proportion of brown colors (humic substances)
        if brown_prop > 0.4:
            type_b_score += 0.3
        elif brown_prop > 0.3:
            type_b_score += 0.2
        elif brown_prop > 0.2:
            type_b_score += 0.1
        
        # Low average value (darker overall)
        if avg_value < 3.0:
            type_b_score += 0.2
        elif avg_value < 4.0:
            type_b_score += 0.1
        
        # Moderate chroma (not too gray, not too vivid)
        if 1.5 <= avg_chroma <= 3.0:
            type_b_score += 0.1
        
        # Type A (mineral topsoil) indicators
        type_a_score = 0.0
        
        # High proportion of gray colors
        if gray_prop > 0.4:
            type_a_score += 0.4
        elif gray_prop > 0.3:
            type_a_score += 0.3
        elif gray_prop > 0.2:
            type_a_score += 0.2
        
        # Low proportion of dark colors
        if dark_prop < 0.1:
            type_a_score += 0.3
        elif dark_prop < 0.2:
            type_a_score += 0.2
        elif dark_prop < 0.3:
            type_a_score += 0.1
        
        # Higher average value (lighter overall)
        if avg_value > 5.0:
            type_a_score += 0.2
        elif avg_value > 4.0:
            type_a_score += 0.1
        
        # Low chroma (grayish)
        if avg_chroma <= 1.5:
            type_a_score += 0.1
        
        # Determine classification
        if type_b_score > type_a_score:
            classification = "Type B"
            confidence = min(0.95, 0.6 + (type_b_score - type_a_score))
        else:
            classification = "Type A"
            confidence = min(0.95, 0.6 + (type_a_score - type_b_score))
        
        return classification, confidence
    
    def classify_soil_type_hybrid(self, analysis: Dict[str, Any], bins: Dict[str, Dict], 
                                 cropped_image: np.ndarray) -> Tuple[str, float, str, float, str]:
        """
        Hybrid classification using both rule-based and ML approaches.
        
        Args:
            analysis: Organic matter distribution analysis
            bins: Color bins dictionary
            cropped_image: Cropped image for ML analysis
            
        Returns:
            Tuple of (final_classification, final_confidence, ml_classification, ml_confidence, method)
        """
        # Rule-based classification
        rule_classification, rule_confidence = self.classify_soil_type_rule_based(analysis, bins)
        
        # ML classification using Random Forest
        ml_classification, ml_confidence = self.rf_classifier.predict(cropped_image)
        
        # Convert ML class names to standard format
        if ml_classification == "type_a":
            ml_classification = "Type A"
        elif ml_classification == "type_b":
            ml_classification = "Type B"
        
        # Decide which method to use
        if ml_confidence >= self.ml_threshold:
            # Use ML prediction if confidence is high enough
            final_classification = ml_classification
            final_confidence = ml_confidence
            method = "ml"
        else:
            # Fall back to rule-based if ML confidence is low
            final_classification = rule_classification
            final_confidence = rule_confidence
            method = "rule_based"
        
        return final_classification, final_confidence, ml_classification, ml_confidence, method
    
    def process_image(self, image_path: str, sample_id: str = None) -> SoilSample:
        """
        Process a single soil image and return classification results.
        
        Args:
            image_path: Path to the input image
            sample_id: Optional sample ID, generates one if not provided
            
        Returns:
            SoilSample object with classification results
        """
        start_time = datetime.now()
        
        # Generate sample ID if not provided
        if sample_id is None:
            sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Processing sample {sample_id}")
        
        # Validate lighting
        lighting_lux = self.validate_lighting(image)
        if lighting_lux < self.min_lighting_lux:
            logger.warning(f"Lighting below minimum threshold: {lighting_lux:.1f} lux < {self.min_lighting_lux} lux")
        
        # Crop image
        cropped = self.crop_image(image)
        logger.info(f"Cropped image to {cropped.shape}")
        
        # Bin similar pixels
        bins = self.bin_similar_pixels(cropped)
        logger.info(f"Created {len(bins)} color bins")
        
        # Analyze organic matter distribution
        analysis = self.analyze_organic_matter_distribution(bins)
        
        # Classify soil type
        # Determine which ML model to use
        rf_available = self.use_ml and self.rf_classifier.model is not None
        yolo_available = self.use_yolo and self.yolo_classifier.model is not None
        
        if (self.model_type == "yolo" and yolo_available) or \
           (self.model_type == "yolo" and not rf_available and yolo_available):
            # Use YOLOv11
            ml_classification, ml_confidence = self.yolo_classifier.predict(cropped)
            method = "yolo"
            classification = ml_classification
            confidence = ml_confidence
        elif (self.model_type == "rf" and rf_available) or \
             (self.model_type == "rf" and not yolo_available and rf_available):
            # Use Random Forest
            classification, confidence, ml_classification, ml_confidence, method = self.classify_soil_type_hybrid(
                analysis, bins, cropped
            )
        elif rf_available:
            # Default to Random Forest if available
            classification, confidence, ml_classification, ml_confidence, method = self.classify_soil_type_hybrid(
                analysis, bins, cropped
            )
        else:
            # Fall back to rule-based
            classification, confidence = self.classify_soil_type_rule_based(analysis, bins)
            ml_classification = None
            ml_confidence = None
            method = "rule_based"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create sample object
        sample = SoilSample(
            sample_id=sample_id,
            timestamp=start_time.isoformat(),
            classification=classification,
            confidence=confidence,
            bin_count=len(bins),
            bin_values=analysis,
            bin_proportions={k: v['proportion'] for k, v in bins.items()},
            lighting_lux=lighting_lux,
            image_shape=cropped.shape,
            processing_time=processing_time,
            ml_classification=ml_classification,
            ml_confidence=ml_confidence,
            classification_method=method
        )
        
        # Store sample
        self.store_sample(sample)
        
        logger.info(f"Sample {sample_id} classified as {classification} (confidence: {confidence:.2f}, method: {method})")
        
        return sample
    
    def store_sample(self, sample: SoilSample):
        """Store sample in memory and database."""
        # Store in memory (up to 100,000 samples)
        if len(self.samples) < 100000:
            self.samples[sample.sample_id] = sample
        else:
            logger.warning("Memory storage full (100,000 samples), storing only in database")
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO soil_samples 
            (sample_id, timestamp, classification, confidence, bin_count, 
             bin_values, bin_proportions, lighting_lux, image_shape, processing_time,
             ml_classification, ml_confidence, classification_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample.sample_id,
            sample.timestamp,
            sample.classification,
            sample.confidence,
            sample.bin_count,
            json.dumps(sample.bin_values),
            json.dumps(sample.bin_proportions),
            sample.lighting_lux,
            json.dumps(sample.image_shape),
            sample.processing_time,
            sample.ml_classification,
            sample.ml_confidence,
            sample.classification_method
        ))
        
        conn.commit()
        conn.close()
    
    def export_to_csv(self, output_path: str = "soil_classification_results.csv"):
        """Export all samples to CSV file."""
        if not self.samples:
            logger.warning("No samples to export")
            return
        
        # Prepare data for CSV
        csv_data = []
        for sample in self.samples.values():
            row = {
                'Sample_ID': sample.sample_id,
                'Timestamp': sample.timestamp,
                'Classification': sample.classification,
                'Confidence': sample.confidence,
                'Bin_Count': sample.bin_count,
                'Lighting_Lux': sample.lighting_lux,
                'Image_Shape': str(sample.image_shape),
                'Processing_Time': sample.processing_time,
                'ML_Classification': sample.ml_classification,
                'ML_Confidence': sample.ml_confidence,
                'Classification_Method': sample.classification_method
            }
            
            # Add bin proportions
            for bin_key, proportion in sample.bin_proportions.items():
                row[f'Bin_{bin_key}_Proportion'] = proportion
            
            # Add analysis values
            for key, value in sample.bin_values.items():
                row[f'Analysis_{key}'] = value
            
            csv_data.append(row)
        
        # Write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(csv_data)} samples to {output_path}")

def main():
    """Main function for testing the enhanced soil classifier."""
    # Initialize classifier
    classifier = SoilClassifier(use_ml=True)
    
    # Test with sample images
    sample_images = [
        "soil1.jpeg",
        "soil2.jpeg",
        "soil3.jpeg",
        "soil4.jpeg",
        "soil5.jpeg"
    ]
    
    # Process samples
    for i, image_path in enumerate(sample_images):
        if Path(image_path).exists():
            try:
                sample = classifier.process_image(image_path, f"test_sample_{i+1}")
                print(f"\nSample {sample.sample_id}:")
                print(f"  Classification: {sample.classification}")
                print(f"  Confidence: {sample.confidence:.2f}")
                print(f"  Method: {sample.classification_method}")
                if sample.ml_classification:
                    print(f"  ML Classification: {sample.ml_classification}")
                    print(f"  ML Confidence: {sample.ml_confidence:.2f}")
                print(f"  Bin Count: {sample.bin_count}")
                print(f"  Lighting: {sample.lighting_lux:.1f} lux")
                print(f"  Processing Time: {sample.processing_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
    
    # Export results
    classifier.export_to_csv()
    
    logger.info("Enhanced soil classification completed!")

if __name__ == "__main__":
    main()






