import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
model_path = r"C:\Users\laksh\OneDrive - Mahindra University\Desktop\yolo_wts\weights\best.pt"


# Path to your test image (adjust if needed)
img_path = r"C:\Users\laksh\OneDrive - Mahindra University\Desktop\dataset_img\7661185-v4fc1be38537-6.jpg"# Update this to your actual location

# Load the custom model
model = YOLO(model_path)

# Run inference with low confidence threshold
results = model(img_path, conf=0.05)
results[0].show() 

