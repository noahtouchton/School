# High-Level Dataset Summary

## Overview

The soil classification system uses **132 total images** to train and evaluate models that distinguish between two soil types:
- **Type A:** Mineral topsoil (91 images)
- **Type B:** Organic-rich material (41 images)

---

## Dataset Splits

### How Images Are Split

The 132 images are divided into three groups using an 80/10/10 ratio:

#### 1. **Training Set** (104 images - ~79%)
**Purpose:** Teaching the model how to classify soil images

**Images:**
- 71 Type A (mineral topsoil)
- 33 Type B (organic-rich)

**Used for:**
- Learning patterns that distinguish Type A from Type B
- Adjusting model parameters during training
- Building the classifier's knowledge base

**Location:** `yolo_dataset/train/`

---

#### 2. **Validation Set** (14 images - ~11%)
**Purpose:** Checking training progress and preventing overfitting

**Images:**
- 10 Type A
- 4 Type B

**Used for:**
- Monitoring model performance during training
- Early stopping (stop training when performance plateaus)
- Selecting the best model checkpoint
- Fine-tuning hyperparameters

**Location:** `yolo_dataset/val/`

---

#### 3. **Test Set** (14 images - ~11%)
**Purpose:** Final evaluation on unseen data

**Images:**
- 10 Type A
- 4 Type B

**Used for:**
- Final performance assessment
- Reporting accuracy metrics
- Ensuring the model generalizes to new images
- **These images are never used during training**

**Location:** `yolo_dataset/test/`

---

## What Each Image Contains

### Type A Images (91 total)
- Mineral topsoil samples
- Lighter, grayer soil appearance
- Less organic matter
- Typical agricultural/construction topsoil

### Type B Images (41 total)
- Organic-rich soil samples
- Darker, browner appearance
- Higher organic content
- Compost, peat, or rich humus material

---

## Training Process

1. **Training Phase:**
   - Model sees the 104 training images
   - Learns patterns: "Lighter gray = Type A" vs "Darker brown = Type B"
   - Adjusts weights iteratively

2. **Validation Phase:**
   - Model tested on 14 validation images
   - Performance monitored to prevent overfitting
   - Best model saved based on validation accuracy

3. **Test Phase:**
   - Final evaluation on 14 unseen test images
   - Reports: 92.86% overall accuracy
   - Type A: 100% accuracy (10/10 correct)
   - Type B: 75% accuracy (3/4 correct)

---

## Visual Summary

```
132 Total Images
├── Training (104 images) ← Model learns from these
│   ├── Type A: 71 images
│   └── Type B: 33 images
│
├── Validation (14 images) ← Training progress monitoring
│   ├── Type A: 10 images
│   └── Type B: 4 images
│
└── Test (14 images) ← Final evaluation
    ├── Type A: 10 images
    └── Type B: 4 images
```

---

## Why This Split?

- **80% Training:** Enough data for the model to learn patterns
- **10% Validation:** Detects overfitting during training
- **10% Test:** Unbiased final evaluation

**Important Note:** This is a small dataset, especially for Type B (only 41 images). The model performs well on Type A but struggles slightly with Type B due to class imbalance and limited training data.

---

## Model Performance Summary

Based on the **14 test images** (never seen during training):

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Accuracy** | 92.86% | 13 out of 14 correct |
| **Type A Accuracy** | 100% | All 10 correct |
| **Type B Accuracy** | 75% | 3 out of 4 correct |
| **Type A Precision** | 90.91% | Very few false positives |
| **Type B Precision** | 100% | No false positives |
| **Type A Recall** | 100% | Finds all Type A |
| **Type B Recall** | 75% | Finds 3 out of 4 Type B |

---

## Real-World Usage

When you run `./setup.sh gui` or `./setup.sh demo`:
- Trained model loads (3.0 MB YOLOv11 file)
- New images classified using learned patterns
- Model never "sees" test images during use
- All classifications based on training knowledge

---

## File Locations

```
soil_dataset/          # Original images (132 total)
├── type_a/           # 91 Type A images
└── type_b/           # 41 Type B images

yolo_dataset/         # Processed dataset (same 132 images, resized and split)
├── train/            # 104 images for training
├── val/              # 14 images for validation  
└── test/             # 14 images for final testing
```



