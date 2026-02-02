#!/usr/bin/env python3
"""
Test script for soil classification using sample images.
"""

from soil_classifier import SoilClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_soil_classification():
    """Test the soil classification system with sample images."""
    
    # Initialize classifier
    classifier = SoilClassifier()
    
    # Test with our sample images
    sample_images = [
        "sample_type_a.jpg",
        "sample_type_b.jpg"
    ]
    
    print("Testing Soil Classification System")
    print("=" * 50)
    
    # Process each sample
    for i, image_path in enumerate(sample_images):
        try:
            sample = classifier.process_image(image_path, f"test_{image_path.replace('.jpg', '')}")
            
            print(f"\nSample {sample.sample_id} ({image_path}):")
            print(f"  Classification: {sample.classification}")
            print(f"  Confidence: {sample.confidence:.2f}")
            print(f"  Bin Count: {sample.bin_count}")
            print(f"  Lighting: {sample.lighting_lux:.1f} lux")
            print(f"  Processing Time: {sample.processing_time:.2f} seconds")
            
            # Print analysis details
            print(f"  Analysis Details:")
            for key, value in sample.bin_values.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
                    
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    # Export results
    classifier.export_to_csv("test_results.csv")
    print(f"\nResults exported to test_results.csv")
    
    # Print summary
    print("\nClassification Summary:")
    print("-" * 30)
    for sample_id, sample in classifier.samples.items():
        expected_type = "Type A" if "type_a" in sample_id else "Type B"
        correct = "✓" if sample.classification == expected_type else "✗"
        print(f"{sample_id}: {sample.classification} (Expected: {expected_type}) {correct}")

if __name__ == "__main__":
    test_soil_classification()
