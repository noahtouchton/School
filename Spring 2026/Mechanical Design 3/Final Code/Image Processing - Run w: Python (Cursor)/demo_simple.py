#!/usr/bin/env python3
"""
Simple Console Demo for Soil Classification System
No GUI required - processes sample images and displays results in terminal
"""

from soil_classifier_enhanced import SoilClassifier
import os
from pathlib import Path

def simple_demo():
    """Simple demonstration of soil classification without GUI."""
    
    print("=" * 70)
    print("🌱 Soil Classification System - Console Demo")
    print("=" * 70)
    print()
    
    # Initialize classifier with YOLOv11
    print("Initializing soil classifier with YOLOv11-cls...")
    try:
        classifier = SoilClassifier(model_type="yolo", use_ml=True, use_yolo=True, ml_threshold=0.6)
        print("✅ Classifier initialized")
        print(f"   YOLOv11 Model: {'Available' if classifier.yolo_classifier.model else 'Not Available'}")
        print(f"   Random Forest: {'Available' if classifier.rf_classifier.model else 'Not Available'}")
        print(f"   Model Type: {classifier.model_type}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    # Find sample images
    sample_images = ["soil1.jpeg", "soil2.jpeg", "soil3.jpeg", "soil4.jpeg", "soil5.jpeg"]
    available_images = [img for img in sample_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ No sample images found!")
        print("Please ensure soil1.jpeg through soil5.jpeg are present.")
        return
    
    print(f"Found {len(available_images)} sample image(s)")
    print()
    
    # Expected classifications
    expected = {
        "soil1.jpeg": "Type A",
        "soil2.jpeg": "Type A",
        "soil3.jpeg": "Type B",
        "soil4.jpeg": "Type A",
        "soil5.jpeg": "Type B"
    }
    
    # Process images
    results = []
    correct_count = 0
    
    for i, img_path in enumerate(available_images, 1):
        filename = os.path.basename(img_path)
        expected_type = expected.get(filename, "Unknown")
        
        print(f"[{i}/{len(available_images)}] Processing: {filename}")
        
        try:
            # Classify the image
            sample = classifier.process_image(img_path, f"sample_{i}")
            
            # Check accuracy
            is_correct = sample.classification == expected_type
            if is_correct:
                correct_count += 1
            
            # Display results
            print(f"    Classification: {sample.classification}")
            print(f"    Expected:       {expected_type}")
            print(f"    Match:          {'✅ Correct' if is_correct else '❌ Incorrect'}")
            print(f"    Confidence:     {sample.confidence:.1%}")
            print(f"    Method:         {sample.classification_method}")
            print(f"    Processing:     {sample.processing_time:.2f}s")
            
            results.append(sample)
            print()
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            print()
    
    # Summary
    if results:
        print("=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        accuracy = (correct_count / len(results)) * 100
        avg_confidence = sum(s.confidence for s in results) / len(results)
        avg_time = sum(s.processing_time for s in results) / len(results)
        
        print(f"Samples Processed:    {len(results)}")
        print(f"Accuracy:             {correct_count}/{len(results)} ({accuracy:.1f}%)")
        print(f"Average Confidence:   {avg_confidence:.1%}")
        print(f"Average Time:         {avg_time:.2f}s")
        print()
        
        # Classification breakdown
        yolo_count = sum(1 for s in results if s.classification_method == "yolo")
        ml_count = sum(1 for s in results if s.classification_method == "ml")
        rule_count = sum(1 for s in results if s.classification_method == "rule_based")
        print(f"YOLOv11:              {yolo_count}")
        print(f"Random Forest (ML):   {ml_count}")
        print(f"Rule-based:           {rule_count}")
        print()
        
        # Export results
        try:
            output_file = "demo_results.csv"
            classifier.export_to_csv(output_file)
            print(f"✅ Results exported to: {output_file}")
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
        
        print()
        print("=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70)
    else:
        print("❌ No samples were processed")

if __name__ == "__main__":
    simple_demo()





