#!/usr/bin/env python3
"""
Generate Corrected Metrics Using ONLY Test Set

This script evaluates models using ONLY the held-out test set from yolo_dataset/test/
"""

from ultralytics import YOLO
import joblib
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import json

def extract_features(image_path: str) -> np.ndarray:
    """Extract features for Random Forest."""
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(50)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    for channel in cv2.split(img):
        features.extend([
            np.mean(channel), np.std(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    
    features.extend([
        np.mean(gray), np.std(gray), np.median(gray),
        np.percentile(gray, 25), np.percentile(gray, 75)
    ])
    
    for channel in cv2.split(hsv):
        features.extend([np.mean(channel), np.std(channel)])
    
    for channel in cv2.split(lab):
        features.extend([np.mean(channel), np.std(channel)])
    
    kernel = np.ones((3,3), np.float32) / 9
    blurred = cv2.filter2D(gray, -1, kernel)
    features.extend([
        np.mean(np.abs(gray - blurred)),
        np.std(np.abs(gray - blurred))
    ])
    
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
    
    return np.array(features)

def main():
    """Generate corrected metrics using ONLY test set."""
    
    print("="*80)
    print("CORRECTED METRICS - Using ONLY Held-Out Test Set")
    print("="*80 + "\n")
    
    # Load models
    print("Loading models...")
    try:
        rf_model = joblib.load("soil_classifier_sklearn.pkl")
        print("✓ Random Forest loaded")
    except:
        rf_model = None
        print("✗ Random Forest not available")
    
    try:
        yolo_model = YOLO("soil_classifier_yolo.pt")
        print("✓ YOLOv11 loaded")
    except:
        yolo_model = None
        print("✗ YOLOv11 not available")
    
    # Load ONLY test set
    test_dir = Path("yolo_dataset/test")
    y_true = []
    image_paths = []
    
    for class_idx, class_name in enumerate(["type_a", "type_b"]):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob("*.JPG"))
        print(f"Found {len(images)} {class_name} test images")
        
        for img_path in images:
            y_true.append(class_idx)
            image_paths.append(str(img_path))
    
    print(f"\nTotal test images: {len(y_true)}")
    print(f"Type A: {y_true.count(0)}, Type B: {y_true.count(1)}\n")
    
    # Predict with YOLOv11
    if yolo_model:
        print("Running YOLOv11 predictions on test set...")
        y_pred_yolo = []
        for img_path in image_paths:
            try:
                results = yolo_model(str(img_path), verbose=False)
                pred = results[0]
                predicted_class_idx = pred.probs.top1
                y_pred_yolo.append(predicted_class_idx)
            except:
                y_pred_yolo.append(-1)
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred_yolo)
        accuracy = accuracy_score(y_true, y_pred_yolo)
        
        tp_a = cm[0,0]
        tn_a = cm[1,1]
        fp_a = cm[1,0]
        fn_a = cm[0,1]
        
        tp_b = cm[1,1]
        tn_b = cm[0,0]
        fp_b = cm[0,1]
        fn_b = cm[1,0]
        
        precision_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
        recall_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
        f1_a = 2 * (precision_a * recall_a) / (precision_a + recall_a) if (precision_a + recall_a) > 0 else 0
        
        precision_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        recall_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        f1_b = 2 * (precision_b * recall_b) / (precision_b + recall_b) if (precision_b + recall_b) > 0 else 0
        
        accuracy_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
        accuracy_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        
        print("="*80)
        print("CORRECTED METRICS - YOLOv11-cls on Test Set")
        print("="*80 + "\n")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        print("Type A Metrics:")
        print(f"  Accuracy:  {accuracy_a:.4f} ({accuracy_a*100:.2f}%)")
        print(f"  Precision: {precision_a:.4f}")
        print(f"  Recall:    {recall_a:.4f}")
        print(f"  F1-Score:  {f1_a:.4f}\n")
        print("Type B Metrics:")
        print(f"  Accuracy:  {accuracy_b:.4f} ({accuracy_b*100:.2f}%)")
        print(f"  Precision: {precision_b:.4f}")
        print(f"  Recall:    {recall_b:.4f}")
        print(f"  F1-Score:  {f1_b:.4f}\n")
        print("Confusion Matrix:")
        print(f"  Type A: TP={cm[0,0]}, FN={cm[0,1]}")
        print(f"  Type B: TP={cm[1,1]}, FN={cm[1,0]}")
        
        # Save to JSON
        corrected_metrics = {
            "test_set_size": len(y_true),
            "overall_accuracy": float(accuracy),
            "type_a": {
                "accuracy": float(accuracy_a),
                "precision": float(precision_a),
                "recall": float(recall_a),
                "f1": float(f1_a),
                "true_positives": int(cm[0,0]),
                "false_negatives": int(cm[0,1])
            },
            "type_b": {
                "accuracy": float(accuracy_b),
                "precision": float(precision_b),
                "recall": float(recall_b),
                "f1": float(f1_b),
                "true_positives": int(cm[1,1]),
                "false_negatives": int(cm[1,0])
            },
            "confusion_matrix": cm.tolist()
        }
        
        with open("corrected_metrics.json", 'w') as f:
            json.dump(corrected_metrics, f, indent=2)
        
        print("\n✓ Corrected metrics saved to: corrected_metrics.json")
        print("="*80)
    
    # Predict with Random Forest
    if rf_model:
        print("\nRunning Random Forest predictions on test set...")
        y_pred_rf = []
        for img_path in image_paths:
            try:
                features = extract_features(img_path)
                features = features.reshape(1, -1)
                prediction = rf_model.predict(features)[0]
                y_pred_rf.append(prediction)
            except:
                y_pred_rf.append(-1)
        
        cm_rf = confusion_matrix(y_true, y_pred_rf)
        accuracy_rf = accuracy_score(y_true, y_pred_rf)
        
        print(f"\nRandom Forest - Overall Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
        print(f"Confusion Matrix: Type A: TP={cm_rf[0,0]}, FN={cm_rf[0,1]}; Type B: TP={cm_rf[1,1]}, FN={cm_rf[1,0]}")
        print("="*80)

if __name__ == "__main__":
    main()




