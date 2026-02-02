#!/usr/bin/env python3
"""
Model Comparison Script

This script evaluates and compares Random Forest and YOLOv11-cls models
for soil classification, generating side-by-side metrics and visualizations.
"""

import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from typing import Dict, List, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    """Compares Random Forest and YOLOv11-cls models."""
    
    def __init__(self, rf_model_path="soil_classifier_sklearn.pkl", 
                 yolo_model_path="soil_classifier_yolo.pt",
                 dataset_dir="soil_dataset"):
        """
        Initialize the comparator.
        
        Args:
            rf_model_path: Path to Random Forest model
            yolo_model_path: Path to YOLOv11 model
            dataset_dir: Directory containing test images
        """
        self.rf_model_path = Path(rf_model_path)
        self.yolo_model_path = Path(yolo_model_path)
        self.dataset_dir = Path(dataset_dir)
        self.rf_model = None
        self.yolo_model = None
        self.class_names = ["type_a", "type_b"]
        
    def load_models(self):
        """Load both models."""
        # Load Random Forest
        if self.rf_model_path.exists():
            try:
                self.rf_model = joblib.load(self.rf_model_path)
                logger.info("Random Forest model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Random Forest model: {e}")
        else:
            logger.warning("Random Forest model not found")
        
        # Load YOLOv11
        if self.yolo_model_path.exists():
            try:
                self.yolo_model = YOLO(str(self.yolo_model_path))
                logger.info("YOLOv11 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv11 model: {e}")
        else:
            logger.warning("YOLOv11 model not found")
    
    def extract_features_rf(self, image_path: str) -> np.ndarray:
        """Extract features for Random Forest (same as training)."""
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(50)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # RGB statistics
        for channel in cv2.split(img):
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
        
        # Texture features
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
    
    def load_test_dataset(self):
        """Load test dataset."""
        logger.info("Loading test dataset...")
        
        y_true = []
        image_paths = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            
            for img_path in images:
                y_true.append(class_idx)
                image_paths.append(str(img_path))
        
        logger.info(f"Loaded {len(y_true)} test images")
        logger.info(f"Type A: {y_true.count(0)}, Type B: {y_true.count(1)}")
        
        return y_true, image_paths
    
    def predict_rf(self, image_path: str) -> int:
        """Predict using Random Forest."""
        if self.rf_model is None:
            return -1
        
        features = self.extract_features_rf(image_path)
        features = features.reshape(1, -1)
        prediction = self.rf_model.predict(features)[0]
        return prediction
    
    def predict_yolo(self, image_path: str) -> int:
        """Predict using YOLOv11."""
        if self.yolo_model is None:
            return -1
        
        try:
            results = self.yolo_model(str(image_path), verbose=False)
            pred = results[0]
            predicted_class_idx = pred.probs.top1
            return predicted_class_idx
        except Exception as e:
            logger.error(f"YOLOv11 prediction failed: {e}")
            return -1
    
    def evaluate_model(self, y_true: List[int], y_pred: List[int], model_name: str) -> Dict:
        """Evaluate a model and return metrics."""
        accuracy = sum(y == p for y, p in zip(y_true, y_pred)) / len(y_true) if y_pred else 0
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        tp_type_a = cm[0, 0]
        fp_type_a = cm[1, 0]
        fn_type_a = cm[0, 1]
        
        tp_type_b = cm[1, 1]
        fp_type_b = cm[0, 1]
        fn_type_b = cm[1, 0]
        
        precision_a = tp_type_a / (tp_type_a + fp_type_a) if (tp_type_a + fp_type_a) > 0 else 0
        recall_a = tp_type_a / (tp_type_a + fn_type_a) if (tp_type_a + fn_type_a) > 0 else 0
        f1_a = 2 * (precision_a * recall_a) / (precision_a + recall_a) if (precision_a + recall_a) > 0 else 0
        
        precision_b = tp_type_b / (tp_type_b + fp_type_b) if (tp_type_b + fp_type_b) > 0 else 0
        recall_b = tp_type_b / (tp_type_b + fn_type_b) if (tp_type_b + fn_type_b) > 0 else 0
        f1_b = 2 * (precision_b * recall_b) / (precision_b + recall_b) if (precision_b + recall_b) > 0 else 0
        
        metrics = {
            'model': model_name,
            'overall_accuracy': accuracy,
            'type_a': {
                'precision': precision_a,
                'recall': recall_a,
                'f1': f1_a
            },
            'type_b': {
                'precision': precision_b,
                'recall': recall_b,
                'f1': f1_b
            },
            'confusion_matrix': cm
        }
        
        return metrics
    
    def compare_models(self):
        """Compare both models."""
        logger.info("Starting model comparison...")
        
        # Load models
        self.load_models()
        
        if not self.rf_model and not self.yolo_model:
            logger.error("No models loaded!")
            return
        
        # Load test dataset
        y_true, image_paths = self.load_test_dataset()
        
        # Predict with both models
        y_pred_rf = []
        y_pred_yolo = []
        
        logger.info("Running predictions...")
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i + 1}/{len(image_paths)} images...")
            
            if self.rf_model:
                pred_rf = self.predict_rf(img_path)
                y_pred_rf.append(pred_rf)
            
            if self.yolo_model:
                pred_yolo = self.predict_yolo(img_path)
                y_pred_yolo.append(pred_yolo)
        
        # Evaluate both models
        results = {}
        
        if self.rf_model:
            results['rf'] = self.evaluate_model(y_true, y_pred_rf, "Random Forest")
        
        if self.yolo_model:
            results['yolo'] = self.evaluate_model(y_true, y_pred_yolo, "YOLOv11-cls")
        
        return results
    
    def print_comparison(self, results: Dict):
        """Print comparison results."""
        print("\n" + "="*80)
        print("MODEL COMPARISON: RANDOM FOREST vs YOLOv11-CLS")
        print("="*80 + "\n")
        
        if 'rf' in results:
            rf = results['rf']
            print("Random Forest:")
            print("-"*80)
            print(f"Overall Accuracy: {rf['overall_accuracy']:.3f} ({rf['overall_accuracy']*100:.1f}%)\n")
            print(f"Type A - Precision: {rf['type_a']['precision']:.3f}, Recall: {rf['type_a']['recall']:.3f}, F1: {rf['type_a']['f1']:.3f}")
            print(f"Type B - Precision: {rf['type_b']['precision']:.3f}, Recall: {rf['type_b']['recall']:.3f}, F1: {rf['type_b']['f1']:.3f}\n")
        
        if 'yolo' in results:
            yolo = results['yolo']
            print("YOLOv11-cls:")
            print("-"*80)
            print(f"Overall Accuracy: {yolo['overall_accuracy']:.3f} ({yolo['overall_accuracy']*100:.1f}%)\n")
            print(f"Type A - Precision: {yolo['type_a']['precision']:.3f}, Recall: {yolo['type_a']['recall']:.3f}, F1: {yolo['type_a']['f1']:.3f}")
            print(f"Type B - Precision: {yolo['type_b']['precision']:.3f}, Recall: {yolo['type_b']['recall']:.3f}, F1: {yolo['type_b']['f1']:.3f}\n")
        
        if 'rf' in results and 'yolo' in results:
            rf = results['rf']
            yolo = results['yolo']
            
            print("COMPARISON SUMMARY:")
            print("-"*80)
            accuracy_diff = yolo['overall_accuracy'] - rf['overall_accuracy']
            print(f"Accuracy Difference: {accuracy_diff:+.3f} ({accuracy_diff*100:+.1f}%)\n")
            
            print("Type A Improvements:")
            print(f"  Precision: {yolo['type_a']['precision'] - rf['type_a']['precision']:+.3f}")
            print(f"  Recall: {yolo['type_a']['recall'] - rf['type_a']['recall']:+.3f}")
            print(f"  F1-Score: {yolo['type_a']['f1'] - rf['type_a']['f1']:+.3f}\n")
            
            print("Type B Improvements:")
            print(f"  Precision: {yolo['type_b']['precision'] - rf['type_b']['precision']:+.3f}")
            print(f"  Recall: {yolo['type_b']['recall'] - rf['type_b']['recall']:+.3f}")
            print(f"  F1-Score: {yolo['type_b']['f1'] - rf['type_b']['f1']:+.3f}\n")
        
        print("="*80)
    
    def visualize_comparison(self, results: Dict, save_path="model_comparison.png"):
        """Visualize comparison of both models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'rf' in results:
            ax1 = axes[0]
            cm_rf = results['rf']['confusion_matrix']
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=ax1)
            ax1.set_title('Random Forest\nAccuracy: {:.1f}%'.format(results['rf']['overall_accuracy']*100))
            ax1.set_ylabel('Actual Class')
            ax1.set_xlabel('Predicted Class')
        
        if 'yolo' in results:
            ax2 = axes[1]
            cm_yolo = results['yolo']['confusion_matrix']
            sns.heatmap(cm_yolo, annot=True, fmt='d', cmap='Greens',
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=ax2)
            ax2.set_title('YOLOv11-cls\nAccuracy: {:.1f}%'.format(results['yolo']['overall_accuracy']*100))
            ax2.set_ylabel('Actual Class')
            ax2.set_xlabel('Predicted Class')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to: {save_path}")
        plt.close()
    
    def save_comparison_report(self, results: Dict, output_path="model_comparison_report.txt"):
        """Save comparison report to file."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SOIL CLASSIFICATION MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            if 'rf' in results:
                rf = results['rf']
                f.write("Random Forest:\n")
                f.write("-"*80 + "\n")
                f.write(f"Overall Accuracy: {rf['overall_accuracy']:.3f} ({rf['overall_accuracy']*100:.1f}%)\n\n")
                f.write(f"Type A - Precision: {rf['type_a']['precision']:.3f}, Recall: {rf['type_a']['recall']:.3f}, F1: {rf['type_a']['f1']:.3f}\n")
                f.write(f"Type B - Precision: {rf['type_b']['precision']:.3f}, Recall: {rf['type_b']['recall']:.3f}, F1: {rf['type_b']['f1']:.3f}\n\n")
            
            if 'yolo' in results:
                yolo = results['yolo']
                f.write("YOLOv11-cls:\n")
                f.write("-"*80 + "\n")
                f.write(f"Overall Accuracy: {yolo['overall_accuracy']:.3f} ({yolo['overall_accuracy']*100:.1f}%)\n\n")
                f.write(f"Type A - Precision: {yolo['type_a']['precision']:.3f}, Recall: {yolo['type_a']['recall']:.3f}, F1: {yolo['type_a']['f1']:.3f}\n")
                f.write(f"Type B - Precision: {yolo['type_b']['precision']:.3f}, Recall: {yolo['type_b']['recall']:.3f}, F1: {yolo['type_b']['f1']:.3f}\n\n")
        
        logger.info(f"Comparison report saved to: {output_path}")

def main():
    """Main function to compare models."""
    comparator = ModelComparator()
    
    logger.info("Starting model comparison...")
    results = comparator.compare_models()
    
    if not results:
        logger.error("Comparison failed!")
        return
    
    # Print comparison
    comparator.print_comparison(results)
    
    # Visualize
    comparator.visualize_comparison(results)
    
    # Save report
    comparator.save_comparison_report(results)
    
    print("\n✅ Model comparison completed!")
    print("📊 Report saved to: model_comparison_report.txt")
    print("📈 Visualization saved to: model_comparison.png")

if __name__ == "__main__":
    main()




