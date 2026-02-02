#!/usr/bin/env python3
"""
Model Evaluation Script

This script generates detailed evaluation metrics for the Random Forest soil classifier
including class-wise accuracy, precision, recall, F1-score, and confusion matrix.
"""

import os
import numpy as np
import cv2
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates soil classification models."""
    
    def __init__(self, model_path="soil_classifier_sklearn.pkl", dataset_dir="soil_dataset"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            dataset_dir: Directory containing type_a and type_b images
        """
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir)
        self.model = None
        self.class_names = ["type_a", "type_b"]
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from a soil image (same as training).
        
        Args:
            image_path: Path to the image
            
        Returns:
            Feature vector
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return np.zeros(50)  # Return zero vector if image can't be loaded
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # Color statistics (RGB)
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
    
    def load_test_dataset(self, test_split_dir=None):
        """
        Load test dataset from soil_dataset directory.
        
        Args:
            test_split_dir: Optional path to a specific test split directory
            
        Returns:
            Tuple of (features, labels, image_paths)
        """
        logger.info("Loading test dataset...")
        
        features = []
        labels = []
        image_paths = []
        
        # If yolo_dataset exists with test split, use that
        if test_split_dir is None and Path("yolo_dataset/test").exists():
            test_split_dir = Path("yolo_dataset/test")
            logger.info(f"Using YOLO test dataset: {test_split_dir}")
        
        if test_split_dir and Path(test_split_dir).exists():
            # Load from organized test split
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = Path(test_split_dir) / class_name
                if not class_dir.exists():
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                
                for img_path in images:
                    feature_vector = self.extract_features(str(img_path))
                    features.append(feature_vector)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))
                
                logger.info(f"Loaded {len(images)} {class_name} images from test set")
        else:
            # Load from original soil_dataset directory (all images)
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = self.dataset_dir / class_name
                if not class_dir.exists():
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                
                logger.info(f"Processing {len(images)} {class_name} images...")
                
                for img_path in images:
                    feature_vector = self.extract_features(str(img_path))
                    features.append(feature_vector)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(features)} test samples")
        logger.info(f"Class distribution: Type A={labels.tolist().count(0)}, Type B={labels.tolist().count(1)}")
        
        return features, labels, image_paths
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and generate metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return None
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Per-class metrics
        metrics = {
            'overall_accuracy': accuracy,
            'class_metrics': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            metrics['class_metrics'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Print evaluation metrics in a readable format."""
        if metrics is None:
            return
        
        print("\n" + "="*70)
        print("SOIL CLASSIFICATION MODEL EVALUATION")
        print("="*70 + "\n")
        
        # Overall accuracy
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)\n")
        
        # Per-class metrics
        print("Per-Class Metrics:")
        print("-"*70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*70)
        
        for class_name, class_metrics in metrics['class_metrics'].items():
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:.3f}         "
                  f"{class_metrics['recall']:.3f}         "
                  f"{class_metrics['f1_score']:.3f}         "
                  f"{int(class_metrics['support']):<10}")
        
        print("-"*70 + "\n")
        
        # Confusion matrix
        print("Confusion Matrix:")
        print("-"*70)
        print(f"{'':<15} {'Predicted Type A':<20} {'Predicted Type B':<20}")
        print("-"*70)
        
        cm = metrics['confusion_matrix']
        print(f"{'Actual Type A':<15} {cm[0,0]:<20} {cm[0,1]:<20}")
        print(f"{'Actual Type B':<15} {cm[1,0]:<20} {cm[1,1]:<20}")
        print("-"*70 + "\n")
    
    def visualize_confusion_matrix(self, metrics: Dict[str, Any], save_path="confusion_matrix.png"):
        """Visualize and save the confusion matrix."""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Soil Classification Confusion Matrix\n(Random Forest Model)')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def save_metrics_to_file(self, metrics: Dict[str, Any], output_path="model_evaluation_metrics.txt"):
        """Save metrics to a text file."""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SOIL CLASSIFICATION MODEL EVALUATION\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-"*70 + "\n")
            
            for class_name, class_metrics in metrics['class_metrics'].items():
                f.write(f"{class_name:<15} "
                       f"{class_metrics['precision']:.3f}         "
                       f"{class_metrics['recall']:.3f}         "
                       f"{class_metrics['f1_score']:.3f}         "
                       f"{int(class_metrics['support']):<10}\n")
            
            f.write("\n" + "-"*70 + "\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*70 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"{'':<15} {'Predicted Type A':<20} {'Predicted Type B':<20}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Actual Type A':<15} {cm[0,0]:<20} {cm[0,1]:<20}\n")
            f.write(f"{'Actual Type B':<15} {cm[1,0]:<20} {cm[1,1]:<20}\n")
            f.write("-"*70 + "\n")
        
        logger.info(f"Metrics saved to: {output_path}")

def main():
    """Main function to evaluate the model."""
    evaluator = ModelEvaluator()
    
    # Load model
    if not evaluator.load_model():
        logger.error("Failed to load model. Please train a model first.")
        return
    
    # Load test dataset
    X_test, y_test, image_paths = evaluator.load_test_dataset()
    
    if len(X_test) == 0:
        logger.error("No test data available!")
        return
    
    # Evaluate
    metrics = evaluator.evaluate_model(X_test, y_test)
    
    if metrics is None:
        logger.error("Evaluation failed!")
        return
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Visualize confusion matrix
    evaluator.visualize_confusion_matrix(metrics)
    
    # Save metrics to file
    evaluator.save_metrics_to_file(metrics)
    
    print("\n✅ Evaluation completed!")
    print("📊 Metrics saved to: model_evaluation_metrics.txt")
    print("📈 Confusion matrix saved to: confusion_matrix.png")

if __name__ == "__main__":
    main()




