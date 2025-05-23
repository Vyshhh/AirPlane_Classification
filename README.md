# Aircraft Classification and Detection Pipeline

This project provides an end-to-end solution for classifying and detecting aircraft types—**civilian**, **military**, and **unmanned aerial vehicles (UAVs)**—from images and videos. It combines **YOLOv8-based annotation**, **CNN-based classification (EfficientNet & ResNet)**, and a planned **hybrid re-training pipeline** for UAV detection.  
Our dataset has 3 classes(Civilian, military, UAV) and each consisting of approx 60k images. Since we have a very large dataset we used DGXA100 GPU. 

## Preprocessing Pipeline

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

## Front End

Our Frontend of "Airplane Classifier using Deep Learning Models" classifies aircraft into civilian, military, and drone categories. The system features a user-friendly web interface built with React, HTML and CSS, allowing users to upload images through an intuitive form with loading indicators and immediate feedback. When images are submitted, they're processed via FormData, sent to the backend using axios, and the deep learning model's predictions are displayed alongside the original image with error handling for a seamless experience that combines technical depth with practical usability.

## Back End

The backend is built using Node.js and Express to handle HTTP requests and manage server-side logic. It uses Multer middleware to receive and store the uploaded image in the server’s uploads/ directory. Once the image is saved, the backend triggers a Python script of ML model (EfficientNet) passing the image path as input. The prediction result, along with the image path and timestamp, is saved into a MongoDB collection using Mongoose. Finally, the backend sends the prediction data back to the frontend in a structured JSON response.

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

## Video Link for Website Demo and Presentation
[Website Demo and Presentation] (https://www.youtube.com/@sreejap1235)
