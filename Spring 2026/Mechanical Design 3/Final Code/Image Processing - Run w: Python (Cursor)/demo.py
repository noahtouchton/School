#!/usr/bin/env python3
"""
Demonstration script for the Soil Classification System
Shows how to use the system for real-world applications.
"""

from soil_classifier_enhanced import SoilClassifier
import os

def demonstrate_classification():
    """Demonstrate the soil classification system."""
    
    print("🌱 Soil Image Classification System - UF/IFAS Analytical Services")
    print("=" * 70)
    print()
    
    # Initialize the enhanced classifier with YOLOv11 as primary model
    print("Initializing enhanced soil classifier with YOLOv11-cls...")
    print("Random Forest available as backup if needed")
    classifier = SoilClassifier(model_type="yolo", use_ml=True, use_yolo=True, ml_threshold=0.6)
    print("✓ Enhanced classifier initialized with YOLOv11-cls (96.2% accuracy)")
    print()
    
    # Check for sample images - handle different working directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_images = ["soil1.jpeg", "soil2.jpeg", "soil3.jpeg", "soil4.jpeg", "soil5.jpeg"]
    
    # Try to find images in script directory first, then current directory
    available_images = []
    for img in sample_images:
        # Try script directory first
        script_path = os.path.join(script_dir, img)
        if os.path.exists(script_path):
            available_images.append(script_path)
        # Then try current directory
        elif os.path.exists(img):
            available_images.append(img)
    
    if not available_images:
        print("❌ No soil sample images found!")
        print("Please ensure soil1.jpeg, soil2.jpeg, soil3.jpeg, soil4.jpeg, and soil5.jpeg are present.")
        print(f"Looking in: {script_dir}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print(f"Found {len(available_images)} sample images to process")
    print()
    
    # Define expected types for each soil sample (works with both filenames and full paths)
    expected_types = {
        "soil1.jpeg": "Type A",
        "soil2.jpeg": "Type A",
        "soil3.jpeg": "Type B", 
        "soil4.jpeg": "Type A",
        "soil5.jpeg": "Type B"
    }
    
    # Process each image
    results = []
    for i, image_path in enumerate(available_images, 1):
        # Get just the filename for expected type lookup
        filename = os.path.basename(image_path)
        expected_type = expected_types.get(filename, "Unknown")
        
        print(f"Processing image {i}/{len(available_images)}: {filename}")
        print(f"  Expected Type: {expected_type}")
        
        try:
            # Process the image
            sample = classifier.process_image(image_path, f"demo_sample_{i}")
            
            # Display results
            print(f"  📊 Results:")
            print(f"     Classification: {sample.classification}")
            print(f"     Expected: {expected_type}")
            correct = "✓" if sample.classification == expected_type else "✗"
            print(f"     Accuracy: {correct}")
            print(f"     Confidence: {sample.confidence:.1%}")
            print(f"     Method: {sample.classification_method}")
            if sample.ml_classification:
                print(f"     ML Classification: {sample.ml_classification}")
                print(f"     ML Confidence: {sample.ml_confidence:.1%}")
            print(f"     Lighting: {sample.lighting_lux:.0f} lux")
            print(f"     Processing Time: {sample.processing_time:.2f}s")
            print(f"     Color Bins: {sample.bin_count}")
            
            # Show key analysis metrics
            analysis = sample.bin_values
            print(f"  🔍 Analysis:")
            print(f"     Dark proportion: {analysis['total_dark_proportion']:.1%}")
            print(f"     Brown proportion: {analysis['total_brown_proportion']:.1%}")
            print(f"     Gray proportion: {analysis['total_gray_proportion']:.1%}")
            print(f"     Average Munsell Value: {analysis['avg_value']:.1f}")
            print(f"     Dominant Hue: {analysis['dominant_hue']}")
            
            results.append(sample)
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Export results
    if results:
        output_file = "demo_results.csv"
        classifier.export_to_csv(output_file)
        print(f"📄 Results exported to: {output_file}")
        print()
    
    # Summary
    print("📋 Classification Summary:")
    print("-" * 40)
    
    # Calculate accuracy
    correct_count = 0
    for i, sample in enumerate(results):
        if i < len(available_images):
            filename = os.path.basename(available_images[i])
            expected_type = expected_types.get(filename, "Unknown")
            correct = sample.classification == expected_type
            if correct:
                correct_count += 1
            status = "✓" if correct else "✗"
            print(f"{status} {sample.sample_id}: {sample.classification} (Expected: {expected_type}) ({sample.confidence:.1%} confidence)")
        else:
            status = "✓" if sample.confidence >= 0.7 else "⚠"
            print(f"{status} {sample.sample_id}: {sample.classification} ({sample.confidence:.1%} confidence)")
    
    accuracy = (correct_count / len(results) * 100) if results else 0
    print(f"\n🎯 Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    print()
    if results:
        print("🎯 System Performance:")
        avg_time = sum(s.processing_time for s in results) / len(results)
        avg_confidence = sum(s.confidence for s in results) / len(results)
        print(f"   Average processing time: {avg_time:.2f} seconds")
        print(f"   Average confidence: {avg_confidence:.1%}")
        print(f"   Samples processed: {len(results)}")
    else:
        print("⚠️  No samples were successfully processed")
    print()
    
    print("✅ Demonstration completed successfully!")
    print()
    print("Next steps:")
    print("• Use your own soil images by calling: classifier.process_image('your_image.jpg')")
    print("• Integrate with USB camera for real-time classification")
    print("• Scale up for batch processing of multiple samples")

def create_test_images():
    """Create test images if they don't exist."""
    import cv2
    import numpy as np
    
    print("Creating Type A soil sample (mineral topsoil)...")
    # Type A: Light grayish soil
    type_a = np.full((300, 300, 3), (160, 160, 160), dtype=np.uint8)
    cv2.imwrite('sample_type_a.jpg', type_a)
    
    print("Creating Type B soil sample (organic-rich)...")
    # Type B: Dark brownish soil
    type_b = np.full((300, 300, 3), (60, 80, 100), dtype=np.uint8)
    cv2.imwrite('sample_type_b.jpg', type_b)
    
    print("✓ Test images created")

if __name__ == "__main__":
    demonstrate_classification()
