#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLOv11 Soil Classification

This script prepares the soil dataset for YOLOv11-cls training by:
1. Loading images from soil_dataset/type_a/ and soil_dataset/type_b/
2. Resizing images to 224x224 (YOLOv11-cls standard)
3. Creating train/validation/test splits (80/10/10)
4. Organizing data in YOLOv11 format
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODatasetPreparator:
    """Prepares soil dataset for YOLOv11-cls training."""
    
    def __init__(self, source_dir="soil_dataset", output_dir="yolo_dataset", img_size=(224, 224)):
        """
        Initialize the dataset preparator.
        
        Args:
            source_dir: Source directory containing type_a and type_b folders
            output_dir: Output directory for YOLOv11 formatted dataset
            img_size: Target image size (width, height)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # Create output directory structure
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        # Class mappings
        self.classes = ["type_a", "type_b"]
        self.class_to_idx = {"type_a": 0, "type_b": 1}
        
    def load_images(self):
        """Load all images from source directories."""
        logger.info("Loading images from source directories...")
        
        images = []
        labels = []
        
        # Load Type A images (mineral topsoil)
        type_a_dir = self.source_dir / "type_a"
        if type_a_dir.exists():
            type_a_images = list(type_a_dir.glob("*.JPG")) + list(type_a_dir.glob("*.jpg")) + list(type_a_dir.glob("*.jpeg"))
            logger.info(f"Found {len(type_a_images)} Type A images")
            
            for img_path in type_a_images:
                images.append(str(img_path))
                labels.append(0)  # Type A = 0
        else:
            logger.warning(f"Type A directory not found: {type_a_dir}")
        
        # Load Type B images (organic-rich material)
        type_b_dir = self.source_dir / "type_b"
        if type_b_dir.exists():
            type_b_images = list(type_b_dir.glob("*.JPG")) + list(type_b_dir.glob("*.jpg")) + list(type_b_dir.glob("*.jpeg"))
            logger.info(f"Found {len(type_b_images)} Type B images")
            
            for img_path in type_b_images:
                images.append(str(img_path))
                labels.append(1)  # Type B = 1
        else:
            logger.warning(f"Type B directory not found: {type_b_dir}")
        
        logger.info(f"Total images loaded: {len(images)}")
        logger.info(f"Type A: {labels.count(0)}, Type B: {labels.count(1)}")
        
        return images, labels
    
    def resize_image(self, image_path, output_path):
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image_path: Path to input image
            output_path: Path to save resized image
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return False
        
        # Resize image
        resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Save resized image
        cv2.imwrite(str(output_path), resized)
        return True
    
    def create_splits(self, images, labels, test_size=0.1, val_size=0.1, random_state=42):
        """
        Create train/validation/test splits.
        
        Args:
            images: List of image paths
            labels: List of corresponding labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility
        """
        logger.info("Creating train/validation/test splits...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} images")
        logger.info(f"Validation set: {len(X_val)} images")
        logger.info(f"Test set: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_yolo_structure(self):
        """Create YOLOv11 directory structure."""
        logger.info("Creating YOLOv11 directory structure...")
        
        # Create main directories
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create class subdirectories
            for class_name in self.classes:
                (split_dir / class_name).mkdir(exist_ok=True)
    
    def copy_images(self, images, labels, split_name):
        """
        Copy and resize images to appropriate split directory.
        
        Args:
            images: List of image paths
            labels: List of corresponding labels
            split_name: Name of the split (train/val/test)
        """
        logger.info(f"Processing {split_name} set...")
        
        split_dir = self.output_dir / split_name
        success_count = 0
        
        for i, (img_path, label) in enumerate(zip(images, labels)):
            class_name = self.classes[label]
            class_dir = split_dir / class_name
            
            # Generate output filename
            img_name = Path(img_path).name
            # Convert .jpeg to .JPG for consistency
            if img_name.lower().endswith('.jpeg'):
                img_name = img_name[:-5] + '.JPG'
            output_path = class_dir / img_name
            
            # Resize and copy image
            if self.resize_image(img_path, output_path):
                success_count += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images for {split_name}")
        
        logger.info(f"Successfully processed {success_count}/{len(images)} images for {split_name}")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLOv11 training."""
        yaml_content = f"""# Soil Classification Dataset Configuration
path: {self.output_dir.absolute()}
train: train
val: val
test: test

# Classes
nc: 2  # number of classes
names: {self.classes}  # class names
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset configuration: {yaml_path}")
    
    def prepare_dataset(self):
        """Main method to prepare the complete dataset."""
        logger.info("Starting dataset preparation...")
        
        # Load images
        images, labels = self.load_images()
        
        if not images:
            logger.error("No images found! Please check the source directory.")
            return False
        
        # Create splits
        train_data, val_data, test_data = self.create_splits(images, labels)
        
        # Create directory structure
        self.create_yolo_structure()
        
        # Copy images to appropriate directories
        self.copy_images(*train_data, "train")
        self.copy_images(*val_data, "val")
        self.copy_images(*test_data, "test")
        
        # Create dataset configuration
        self.create_dataset_yaml()
        
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Dataset saved to: {self.output_dir.absolute()}")
        
        return True
    
    def print_dataset_summary(self):
        """Print summary of the prepared dataset."""
        logger.info("Dataset Summary:")
        logger.info("=" * 50)
        
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            logger.info(f"\n{split.upper()} SET:")
            
            for class_name in self.classes:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")))
                    logger.info(f"  {class_name}: {count} images")
        
        logger.info("=" * 50)

def main():
    """Main function to prepare the dataset."""
    preparator = YOLODatasetPreparator()
    
    if preparator.prepare_dataset():
        preparator.print_dataset_summary()
        print("\n✅ Dataset preparation completed successfully!")
        print(f"📁 Dataset location: {preparator.output_dir.absolute()}")
        print("\nNext steps:")
        print("1. Install YOLOv11: pip install ultralytics")
        print("2. Train model: yolo train model=yolov11n-cls.pt data=yolo_dataset/dataset.yaml epochs=50 imgsz=224")
    else:
        print("❌ Dataset preparation failed!")

if __name__ == "__main__":
    main()

