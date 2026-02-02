#!/usr/bin/env python3
"""
Scikit-learn Soil Classification Training Script

This script trains a scikit-learn based classifier for soil classification using the prepared dataset.
Since YOLOv11 requires PyTorch which doesn't support Python 3.14 yet, we'll use scikit-learn
as an alternative approach for now.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SklearnSoilClassifier:
    """Scikit-learn based soil classifier."""
    
    def __init__(self, dataset_dir="yolo_dataset", model_save_path="soil_classifier_sklearn.pkl"):
        """
        Initialize the classifier.
        
        Args:
            dataset_dir: Directory containing the prepared dataset
            model_save_path: Path to save the trained model
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_save_path = model_save_path
        self.model = None
        self.class_names = ["type_a", "type_b"]
        self.feature_names = None
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from a soil image.
        
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
        
        # Texture features (simplified)
        # Calculate local binary pattern-like features
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
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the entire dataset and extract features.
        
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Loading dataset and extracting features...")
        
        features = []
        labels = []
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_dir / split
            
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = split_dir / class_name
                
                if not class_dir.exists():
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                # Get all images in this class
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                
                logger.info(f"Processing {len(images)} {class_name} images from {split} set...")
                
                for img_path in images:
                    feature_vector = self.extract_features(img_path)
                    features.append(feature_vector)
                    labels.append(class_idx)
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features each")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return features, labels
    
    def train_model(self, features: np.ndarray, labels: np.ndarray, test_size: 0.2):
        """
        Train the scikit-learn model.
        
        Args:
            features: Feature matrix
            labels: Label vector
            test_size: Proportion of data to use for testing
        """
        logger.info("Training scikit-learn model...")
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Try multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            logger.info(f"{name} - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_name = name
        
        self.model = best_model
        logger.info(f"Best model: {best_name} with test accuracy: {best_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=self.class_names))
        
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        
        return best_score
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            logger.error("No model to save!")
            return False
        
        logger.info(f"Saving model to: {self.model_save_path}")
        joblib.dump(self.model, self.model_save_path)
        logger.info("Model saved successfully!")
        return True
    
    def load_model(self):
        """Load a previously trained model."""
        if not Path(self.model_save_path).exists():
            logger.error(f"Model file not found: {self.model_save_path}")
            return False
        
        logger.info(f"Loading model from: {self.model_save_path}")
        self.model = joblib.load(self.model_save_path)
        logger.info("Model loaded successfully!")
        return True
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict soil type for a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return "Unknown", 0.0
        
        # Extract features
        features = self.extract_features(image_path)
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

def main():
    """Main function to train the scikit-learn model."""
    logger.info("Starting Scikit-learn Soil Classification Training")
    
    # Initialize classifier
    classifier = SklearnSoilClassifier()
    
    # Load dataset
    features, labels = classifier.load_dataset()
    
    if len(features) == 0:
        logger.error("No features extracted! Check dataset.")
        return
    
    # Train model
    accuracy = classifier.train_model(features, labels, test_size=0.2)
    
    # Save model
    classifier.save_model()
    
    logger.info("Training completed successfully!")
    print(f"\n✅ Scikit-learn training completed!")
    print(f"📁 Model saved as: {classifier.model_save_path}")
    print(f"🎯 Test accuracy: {accuracy:.3f}")
    
    # Test the model on a few samples
    print("\n🧪 Testing model on sample images...")
    test_images = [
        "soil1.jpeg",
        "soil2.jpeg", 
        "soil3.jpeg",
        "soil4.jpeg",
        "soil5.jpeg"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            predicted_class, confidence = classifier.predict(img_path)
            print(f"  {img_path}: {predicted_class} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()






