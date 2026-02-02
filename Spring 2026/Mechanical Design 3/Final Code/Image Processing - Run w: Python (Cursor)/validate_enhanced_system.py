#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Soil Classification System

This script validates the enhanced soil classification system by:
1. Testing both rule-based and ML-based classification
2. Comparing accuracy between methods
3. Validating confidence scores
4. Testing on the prepared dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
from soil_classifier_enhanced import SoilClassifier
import logging
import pandas as pd
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoilClassificationValidator:
    """Validates the enhanced soil classification system."""
    
    def __init__(self):
        """Initialize the validator."""
        self.results = []
        self.dataset_dir = Path("yolo_dataset")
        
    def test_sample_images(self):
        """Test on sample images with known classifications."""
        logger.info("Testing on sample images...")
        
        # Expected classifications for sample images
        expected_classifications = {
            "soil1.jpeg": "Type A",
            "soil2.jpeg": "Type A", 
            "soil3.jpeg": "Type B",
            "soil4.jpeg": "Type A",
            "soil5.jpeg": "Type B"
        }
        
        # Test with ML enabled
        classifier_ml = SoilClassifier(use_ml=True, ml_threshold=0.6)
        
        # Test with ML disabled (rule-based only)
        classifier_rules = SoilClassifier(use_ml=False)
        
        sample_results = []
        
        for img_name, expected in expected_classifications.items():
            if not Path(img_name).exists():
                logger.warning(f"Sample image not found: {img_name}")
                continue
            
            logger.info(f"Testing {img_name} (expected: {expected})")
            
            # Test with ML
            try:
                sample_ml = classifier_ml.process_image(img_name, f"ml_{img_name}")
                ml_correct = sample_ml.classification == expected
                
                # Test with rule-based
                sample_rules = classifier_rules.process_image(img_name, f"rules_{img_name}")
                rules_correct = sample_rules.classification == expected
                
                result = {
                    'image': img_name,
                    'expected': expected,
                    'ml_classification': sample_ml.classification,
                    'ml_confidence': sample_ml.confidence,
                    'ml_correct': ml_correct,
                    'ml_method': sample_ml.classification_method,
                    'rules_classification': sample_rules.classification,
                    'rules_confidence': sample_rules.confidence,
                    'rules_correct': rules_correct,
                    'rules_method': sample_rules.classification_method
                }
                
                sample_results.append(result)
                
                logger.info(f"  ML: {sample_ml.classification} ({sample_ml.confidence:.3f}) - {'✓' if ml_correct else '✗'}")
                logger.info(f"  Rules: {sample_rules.classification} ({sample_rules.confidence:.3f}) - {'✓' if rules_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"Error testing {img_name}: {e}")
        
        return sample_results
    
    def test_dataset_images(self):
        """Test on the prepared dataset images."""
        logger.info("Testing on dataset images...")
        
        classifier = SoilClassifier(use_ml=True, ml_threshold=0.6)
        dataset_results = []
        
        for split in ['test']:  # Only test on test set
            split_dir = self.dataset_dir / split
            
            for class_name in ['type_a', 'type_b']:
                class_dir = split_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                # Convert class name to expected classification
                expected = "Type A" if class_name == "type_a" else "Type B"
                
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
                
                logger.info(f"Testing {len(images)} {class_name} images from {split} set...")
                
                for img_path in images:
                    try:
                        sample = classifier.process_image(str(img_path), f"dataset_{img_path.name}")
                        correct = sample.classification == expected
                        
                        result = {
                            'image': str(img_path),
                            'expected': expected,
                            'classification': sample.classification,
                            'confidence': sample.confidence,
                            'correct': correct,
                            'method': sample.classification_method,
                            'ml_classification': sample.ml_classification,
                            'ml_confidence': sample.ml_confidence
                        }
                        
                        dataset_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error testing {img_path}: {e}")
        
        return dataset_results
    
    def test_confidence_thresholds(self):
        """Test different confidence thresholds for ML classification."""
        logger.info("Testing different confidence thresholds...")
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = []
        
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold}")
            
            classifier = SoilClassifier(use_ml=True, ml_threshold=threshold)
            
            # Test on sample images
            correct_count = 0
            total_count = 0
            ml_used_count = 0
            
            expected_classifications = {
                "soil1.jpeg": "Type A",
                "soil2.jpeg": "Type A", 
                "soil3.jpeg": "Type B",
                "soil4.jpeg": "Type A",
                "soil5.jpeg": "Type B"
            }
            
            for img_name, expected in expected_classifications.items():
                if not Path(img_name).exists():
                    continue
                
                try:
                    sample = classifier.process_image(img_name, f"threshold_{threshold}_{img_name}")
                    correct = sample.classification == expected
                    
                    if correct:
                        correct_count += 1
                    total_count += 1
                    
                    if sample.classification_method == "ml":
                        ml_used_count += 1
                        
                except Exception as e:
                    logger.error(f"Error testing {img_name} with threshold {threshold}: {e}")
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            ml_usage = ml_used_count / total_count if total_count > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'ml_usage': ml_usage,
                'correct_count': correct_count,
                'total_count': total_count
            })
            
            logger.info(f"  Threshold {threshold}: Accuracy {accuracy:.3f}, ML Usage {ml_usage:.3f}")
        
        return threshold_results
    
    def generate_report(self, sample_results: List[Dict], dataset_results: List[Dict], 
                       threshold_results: List[Dict]):
        """Generate a comprehensive validation report."""
        logger.info("Generating validation report...")
        
        print("\n" + "="*80)
        print("ENHANCED SOIL CLASSIFICATION SYSTEM - VALIDATION REPORT")
        print("="*80)
        
        # Sample image results
        print("\n📊 SAMPLE IMAGE RESULTS:")
        print("-" * 50)
        
        ml_correct = sum(1 for r in sample_results if r['ml_correct'])
        rules_correct = sum(1 for r in sample_results if r['rules_correct'])
        total_samples = len(sample_results)
        
        print(f"ML Classification Accuracy: {ml_correct}/{total_samples} ({ml_correct/total_samples:.1%})")
        print(f"Rule-based Accuracy: {rules_correct}/{total_samples} ({rules_correct/total_samples:.1%})")
        
        print("\nDetailed Results:")
        for result in sample_results:
            ml_status = "✓" if result['ml_correct'] else "✗"
            rules_status = "✓" if result['rules_correct'] else "✗"
            print(f"  {result['image']}:")
            print(f"    Expected: {result['expected']}")
            print(f"    ML: {result['ml_classification']} ({result['ml_confidence']:.3f}) {ml_status}")
            print(f"    Rules: {result['rules_classification']} ({result['rules_confidence']:.3f}) {rules_status}")
        
        # Dataset results
        if dataset_results:
            print("\n📊 DATASET TEST RESULTS:")
            print("-" * 50)
            
            dataset_correct = sum(1 for r in dataset_results if r['correct'])
            dataset_total = len(dataset_results)
            dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0
            
            print(f"Dataset Test Accuracy: {dataset_correct}/{dataset_total} ({dataset_accuracy:.1%})")
            
            # Method breakdown
            ml_method_count = sum(1 for r in dataset_results if r['method'] == 'ml')
            rules_method_count = sum(1 for r in dataset_results if r['method'] == 'rule_based')
            
            print(f"ML Method Used: {ml_method_count}/{dataset_total} ({ml_method_count/dataset_total:.1%})")
            print(f"Rule-based Method Used: {rules_method_count}/{dataset_total} ({rules_method_count/dataset_total:.1%})")
        
        # Threshold analysis
        print("\n📊 CONFIDENCE THRESHOLD ANALYSIS:")
        print("-" * 50)
        
        print("Threshold | Accuracy | ML Usage")
        print("-" * 35)
        for result in threshold_results:
            print(f"   {result['threshold']:.1f}    |   {result['accuracy']:.3f}   |  {result['ml_usage']:.3f}")
        
        # Find optimal threshold
        best_threshold = max(threshold_results, key=lambda x: x['accuracy'])
        print(f"\nOptimal Threshold: {best_threshold['threshold']} (Accuracy: {best_threshold['accuracy']:.3f})")
        
        # Performance summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        avg_ml_confidence = np.mean([r['ml_confidence'] for r in sample_results if r['ml_confidence']])
        avg_rules_confidence = np.mean([r['rules_confidence'] for r in sample_results if r['rules_confidence']])
        
        print(f"Average ML Confidence: {avg_ml_confidence:.3f}")
        print(f"Average Rule-based Confidence: {avg_rules_confidence:.3f}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        if ml_correct >= rules_correct:
            print("✅ ML classification performs as well or better than rule-based")
        else:
            print("⚠️  Rule-based classification outperforms ML - consider retraining")
        
        if best_threshold['threshold'] >= 0.6:
            print(f"✅ Current threshold (0.6) is appropriate")
        else:
            print(f"⚠️  Consider lowering threshold to {best_threshold['threshold']} for better ML usage")
        
        print("\n" + "="*80)
    
    def run_validation(self):
        """Run complete validation suite."""
        logger.info("Starting comprehensive validation...")
        
        # Test sample images
        sample_results = self.test_sample_images()
        
        # Test dataset images
        dataset_results = self.test_dataset_images()
        
        # Test confidence thresholds
        threshold_results = self.test_confidence_thresholds()
        
        # Generate report
        self.generate_report(sample_results, dataset_results, threshold_results)
        
        logger.info("Validation completed successfully!")

def main():
    """Main function to run validation."""
    validator = SoilClassificationValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()






