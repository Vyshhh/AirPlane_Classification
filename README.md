# Aircraft Classification and Detection Pipeline

This project provides an end-to-end solution for classifying and detecting aircraft types—**civilian**, **military**, and **unmanned aerial vehicles (UAVs)**—from images and videos. It combines **YOLOv8-based annotation**, **CNN-based classification (EfficientNet & ResNet)**, and a planned **hybrid re-training pipeline** for UAV detection.

## Overview

Due to the massive size of the aircraft dataset, manual annotation was impractical. We began with **automatic annotation using YOLOv8** (detecting COCO class 4: airplane) and grouped images into three categories. Since YOLO struggled with UAVs, we trained **EfficientNet and ResNet classifiers** which performed well without needing bounding boxes. These classifiers will now assist in labeling UAVs to further **retrain YOLO** for comprehensive detection.

## ⚙Preprocessing Pipeline

### Image Resizing

All images were resized to a uniform shape of **640×640** using OpenCV with **bicubic interpolation**, to match the input size expected by YOLOv8 and improve consistency in training.

```python
def resize_image(img):
    return cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
```

### Noise Handling

Unreadable or corrupted images were skipped during processing. A placeholder empty label file was written in cases where no object was detected.

### Data Augmentation

To improve generalization and reduce overfitting during CNN training, we applied the following augmentation techniques:
* **Random Horizontal Flip**
* **Random Rotation (up to 25°)**

## Model Training

### YOLOv8 Annotation (Auto-Labeling)

We used a pre-trained **YOLOv8x** model to detect airplanes and auto-generate labels in **YOLO format**. Detections with COCO class ID 4 (airplane) were extracted and remapped to:
* `0` → Civilian
* `1` → Military
* `2` → UAV

Each image was saved with bounding boxes drawn, and annotations were written to `.txt` files.

*Output includes:*
* Annotated images (`annotated_train_*`)
* Label files (`labels_train_*`)

Run:
```bash
python annotation.py
```

### EfficientNet Classification

We trained **EfficientNet-B0** with mixed-precision training and early stopping. The final classification layer was modified to match our number of aircraft classes.
* Dropout: `0.5`
* Optimizer: Adam (`lr=5e-5`, `weight_decay=1e-4`)
* Loss: CrossEntropy with label smoothing

```bash
python EfficientNet.py
```

### ResNet-50 Classification

ResNet-50 was fine-tuned by unfreezing only the last block (`layer4`) and the final fully connected layer (`fc`).
* Same augmentations, optimizer, and loss setup
* Early stopping and best model saving implemented

```bash
python ResNet.py
```

### Performance Metrics

**Best model accuracy:**
* **Train:** ~96%
* **Validation:** ~86%

## Results Visualization

The training process includes visualization of:
* Loss curves
* Accuracy metrics
* Confusion matrices 
* Class activation maps

## Hybrid Re-Training Pipeline

Our approach combines the strengths of different models:
1. Use YOLOv8 for initial detection of aircraft
2. Apply classifiers to further categorize detected objects
3. Use classifier decisions to generate additional training data
4. Retrain YOLO with enhanced dataset for end-to-end detection

### Dataset Preparation

Organize your dataset with the following structure:
```
data/
├── train/
│   ├── civilian/
│   ├── military/
│   └── uav/
├── valid/
│   ├── civilian/
│   ├── military/
│   └── uav/
└── test/
    ├── civilian/
    ├── military/
    └── uav/
```

## Future Work

- [ ] Implement ensemble methods combining YOLOv8, EfficientNet, and ResNet
- [ ] Create web interface for real-time aircraft detection
- [ ] Expand dataset with more UAV examples
- [ ] Test model performance on video streams
- [ ] Deploy optimized models on edge devices
