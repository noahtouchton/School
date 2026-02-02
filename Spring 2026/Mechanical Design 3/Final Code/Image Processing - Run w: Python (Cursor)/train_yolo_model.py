#!/usr/bin/env python3
"""
YOLOv11 Soil Classification Training Script

This script trains a YOLOv11-cls model for soil classification using the prepared dataset.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOSoilTrainer:
    """Handles YOLOv11-cls training for soil classification."""
    
    def __init__(self, dataset_yaml="yolo_dataset/dataset.yaml", model_name="yolo11n-cls.pt"):
        """
        Initialize the trainer.
        
        Args:
            dataset_yaml: Path to dataset configuration file
            model_name: YOLOv11 model variant to use
        """
        self.dataset_yaml = Path(dataset_yaml)
        self.model_name = model_name
        self.model = None
        self.results = None
        
    def check_dataset(self):
        """Check if dataset is properly prepared."""
        logger.info("Checking dataset...")
        
        if not self.dataset_yaml.exists():
            logger.error(f"Dataset configuration not found: {self.dataset_yaml}")
            return False
        
        # Load dataset config
        with open(self.dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_path = Path(config['path'])
        
        # Check if directories exist
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if not split_dir.exists():
                logger.error(f"Split directory not found: {split_dir}")
                return False
            
            # Check class directories
            for class_name in config['names']:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.error(f"Class directory not found: {class_dir}")
                    return False
                
                # Count images
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                logger.info(f"{split}/{class_name}: {len(images)} images")
        
        logger.info("Dataset check completed successfully!")
        return True
    
    def load_model(self):
        """Load YOLOv11-cls model."""
        logger.info(f"Loading YOLOv11 model: {self.model_name}")
        
        try:
            # Create model - it will auto-download if needed
            self.model = YOLO(self.model_name, task='classify')
            logger.info("Model created successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def train_model(self, epochs=50, imgsz=224, batch_size=16, patience=10, save_dir="runs/classify"):
        """
        Train the YOLOv11-cls model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
            patience: Early stopping patience
            save_dir: Directory to save training results
        """
        logger.info("Starting model training...")
        logger.info(f"Training parameters:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Patience: {patience}")
        
        try:
            # For YOLOv11-cls, data should be the dataset directory
            dataset_dir = Path(self.dataset_yaml).parent  # Get directory containing dataset.yaml
            logger.info(f"Using dataset directory: {dataset_dir}")
            
            # Train the model
            self.results = self.model.train(
                data=str(dataset_dir),  # Use directory, not YAML file
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                patience=patience,
                project=save_dir,
                name="soil_classification",
                exist_ok=True,
                verbose=True
            )
            
            logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def validate_model(self):
        """Validate the trained model."""
        logger.info("Validating trained model...")
        
        try:
            # Run validation
            val_results = self.model.val()
            
            logger.info("Validation completed!")
            logger.info(f"Validation accuracy: {val_results.top1:.3f}")
            logger.info(f"Validation top5 accuracy: {val_results.top5:.3f}")
            
            return val_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None
    
    def test_model(self, test_images_dir=None):
        """Test the model on test set."""
        logger.info("Testing model on test set...")
        
        try:
            # Load dataset config to get test directory
            with open(self.dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_path = Path(config['path'])
            test_dir = dataset_path / 'test'
            
            # Test on all images in test directory
            test_results = []
            
            for class_name in config['names']:
                class_dir = test_dir / class_name
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                
                logger.info(f"Testing {len(images)} {class_name} images...")
                
                for img_path in images:
                    # Run prediction
                    results = self.model(str(img_path))
                    
                    # Extract prediction
                    pred = results[0]
                    predicted_class = pred.names[pred.probs.top1]
                    confidence = pred.probs.top1conf.item()
                    
                    test_results.append({
                        'image': str(img_path),
                        'true_class': class_name,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'correct': predicted_class == class_name
                    })
            
            # Calculate test accuracy
            correct = sum(1 for r in test_results if r['correct'])
            total = len(test_results)
            accuracy = correct / total if total > 0 else 0
            
            logger.info(f"Test accuracy: {correct}/{total} ({accuracy:.3f})")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            return None
    
    def save_model(self, model_path="soil_classifier_yolo.pt"):
        """Save the trained model."""
        logger.info(f"Saving model to: {model_path}")
        
        try:
            self.model.save(model_path)
            logger.info("Model saved successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def print_training_summary(self):
        """Print training summary."""
        if self.results is None:
            logger.warning("No training results available")
            return
        
        logger.info("Training Summary:")
        logger.info("=" * 50)
        if hasattr(self.results, 'top1'):
            logger.info(f"Top-1 Accuracy: {self.results.top1:.3f}")
        if hasattr(self.results, 'top5'):
            logger.info(f"Top-5 Accuracy: {self.results.top5:.3f}")
        if hasattr(self.results, 'train_time'):
            logger.info(f"Training time: {self.results.train_time:.1f} seconds")
        logger.info("=" * 50)

def main():
    """Main function to train the YOLOv11 model."""
    logger.info("Starting YOLOv11 Soil Classification Training")
    
    # Initialize trainer
    trainer = YOLOSoilTrainer()
    
    # Check dataset
    if not trainer.check_dataset():
        logger.error("Dataset check failed!")
        return
    
    # Load model
    if not trainer.load_model():
        logger.error("Model loading failed!")
        return
    
    # Train model
    if not trainer.train_model(epochs=50, imgsz=224, batch_size=16):
        logger.error("Training failed!")
        return
    
    # Print training summary
    trainer.print_training_summary()
    
    # Validate model
    val_results = trainer.validate_model()
    
    # Test model
    test_results = trainer.test_model()
    
    # Save model
    trainer.save_model()
    
    logger.info("Training pipeline completed successfully!")
    print("\n✅ YOLOv11 training completed!")
    print("📁 Model saved as: soil_classifier_yolo.pt")
    print("📊 Check runs/classify/soil_classification/ for detailed results")

if __name__ == "__main__":
    main()






