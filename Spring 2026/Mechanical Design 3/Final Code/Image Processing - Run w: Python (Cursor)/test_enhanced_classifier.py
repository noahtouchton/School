#!/usr/bin/env python3
"""
Test script for the enhanced soil classifier with YOLOv11 integration.

This script demonstrates the new toggle capability between Random Forest and YOLOv11-cls models.
"""

from soil_classifier_enhanced import SoilClassifier
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_classifier(model_type="yolo"):
    """Test the classifier with a specific model type."""
    print(f"\n{'='*70}")
    print(f"Testing Enhanced Soil Classifier with {model_type.upper()}")
    print(f"{'='*70}\n")
    
    # Initialize classifier
    classifier = SoilClassifier(model_type=model_type, use_yolo=True, use_ml=True)
    
    # Test images
    test_images = [
        "soil1.jpeg",
        "soil2.jpeg",
        "soil3.jpeg",
        "soil4.jpeg",
        "soil5.jpeg"
    ]
    
    results = []
    for img in test_images:
        img_path = Path(img)
        if img_path.exists():
            try:
                sample = classifier.process_image(str(img_path))
                results.append({
                    'image': img,
                    'classification': sample.classification,
                    'confidence': sample.confidence,
                    'method': sample.classification_method
                })
                print(f"{img}:")
                print(f"  Classification: {sample.classification}")
                print(f"  Confidence: {sample.confidence:.2%}")
                print(f"  Method: {sample.classification_method}")
                print()
            except Exception as e:
                logger.error(f"Error processing {img}: {e}")
    
    return results

def main():
    """Main function to test both models."""
    print("\n" + "="*70)
    print("SOIL CLASSIFIER TESTING - YOLOv11 INTEGRATION")
    print("="*70)
    
    # Test with YOLOv11
    yolo_results = test_classifier(model_type="yolo")
    
    # Test with Random Forest
    print("\n" + "="*70)
    rf_results = test_classifier(model_type="rf")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nYOLOv11 Results: {len(yolo_results)} images processed")
    print(f"Random Forest Results: {len(rf_results)} images processed")
    
    if yolo_results and rf_results:
        print("\nComparison:")
        print("-"*70)
        for yolo, rf in zip(yolo_results, rf_results):
            if yolo['classification'] != rf['classification']:
                print(f"{yolo['image']}:")
                print(f"  YOLOv11: {yolo['classification']} ({yolo['confidence']:.2%})")
                print(f"  RF:      {rf['classification']} ({rf['confidence']:.2%})")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()




