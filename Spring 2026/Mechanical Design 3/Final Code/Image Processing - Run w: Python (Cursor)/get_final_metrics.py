#!/usr/bin/env python3
"""
Generate Final Metrics Report

This script generates the final metrics report with Overall Accuracy,
Type A/B Accuracy, Precision, and Recall for the integrated YOLOv11 system.
"""

from compare_models import ModelComparator
import json
from pathlib import Path

def main():
    """Generate final metrics report."""
    print("\n" + "="*80)
    print("GENERATING FINAL METRICS REPORT")
    print("="*80 + "\n")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load models
    comparator.load_models()
    
    if not comparator.rf_model and not comparator.yolo_model:
        print("❌ No models loaded!")
        return
    
    # Load test dataset
    y_true, image_paths = comparator.load_test_dataset()
    
    if len(y_true) == 0:
        print("❌ No test data available!")
        return
    
    # Predict with both models
    print("Running predictions...")
    y_pred_rf = []
    y_pred_yolo = []
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(image_paths)} images...")
        
        if comparator.rf_model:
            pred_rf = comparator.predict_rf(img_path)
            y_pred_rf.append(pred_rf)
        
        if comparator.yolo_model:
            pred_yolo = comparator.predict_yolo(img_path)
            y_pred_yolo.append(pred_yolo)
    
    # Evaluate both models
    results = {}
    
    if comparator.rf_model:
        results['rf'] = comparator.evaluate_model(y_true, y_pred_rf, "Random Forest")
    
    if comparator.yolo_model:
        results['yolo'] = comparator.evaluate_model(y_true, y_pred_yolo, "YOLOv11-cls")
    
    # Generate final metrics report
    print("\n" + "="*80)
    print("FINAL METRICS REPORT")
    print("="*80 + "\n")
    
    # YOLOv11 Metrics
    if 'yolo' in results:
        yolo = results['yolo']
        cm = yolo['confusion_matrix']
        
        print("YOLOv11-cls (PRIMARY MODEL):")
        print("-"*80)
        print(f"Overall Accuracy: {yolo['overall_accuracy']:.4f} ({yolo['overall_accuracy']*100:.2f}%)\n")
        
        print("Type A Metrics:")
        print(f"  Accuracy:  {(cm[0,0] / (cm[0,0] + cm[0,1])):.4f} ({cm[0,0] / (cm[0,0] + cm[0,1])*100:.2f}%)" if (cm[0,0] + cm[0,1]) > 0 else "  Accuracy:  N/A")
        print(f"  Precision: {yolo['type_a']['precision']:.4f}")
        print(f"  Recall:    {yolo['type_a']['recall']:.4f}")
        print(f"  F1-Score:  {yolo['type_a']['f1']:.4f}\n")
        
        print("Type B Metrics:")
        print(f"  Accuracy:  {(cm[1,1] / (cm[1,0] + cm[1,1])):.4f} ({cm[1,1] / (cm[1,0] + cm[1,1])*100:.2f}%)" if (cm[1,0] + cm[1,1]) > 0 else "  Accuracy:  N/A")
        print(f"  Precision: {yolo['type_b']['precision']:.4f}")
        print(f"  Recall:    {yolo['type_b']['recall']:.4f}")
        print(f"  F1-Score:  {yolo['type_b']['f1']:.4f}\n")
        
        # Save to JSON
        metrics_report = {
            "model": "YOLOv11-cls",
            "overall_accuracy": yolo['overall_accuracy'],
            "type_a": {
                "accuracy": (cm[0,0] / (cm[0,0] + cm[0,1])) if (cm[0,0] + cm[0,1]) > 0 else 0,
                "precision": yolo['type_a']['precision'],
                "recall": yolo['type_a']['recall'],
                "f1": yolo['type_a']['f1']
            },
            "type_b": {
                "accuracy": (cm[1,1] / (cm[1,0] + cm[1,1])) if (cm[1,0] + cm[1,1]) > 0 else 0,
                "precision": yolo['type_b']['precision'],
                "recall": yolo['type_b']['recall'],
                "f1": yolo['type_b']['f1']
            },
            "confusion_matrix": cm.tolist()
        }
        
        with open("final_metrics_report.json", 'w') as f:
            json.dump(metrics_report, f, indent=2)
        
        print("="*80)
        print("✅ Metrics saved to: final_metrics_report.json")
        print("="*80)
    else:
        print("❌ YOLOv11 model not available!")

if __name__ == "__main__":
    main()

