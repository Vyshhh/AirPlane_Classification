import os
import cv2
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ---------- Configuration ----------
TARGET_SIZE = (640, 640)
INTERPOLATION = cv2.INTER_CUBIC
BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('/dgxa_home/se22ucse255/se_project/yolov8x.pt').to(device)

# Class mapping - map from COCO class ID to your custom class ID
# In COCO dataset, class ID 4 is "airplane"
# Map all detected airplanes to the appropriate class based on folder
CLASS_MAPPINGS = {
    "civilian_planes": 0,  # Civilian plane class ID
    "military_planes": 1,  # Military plane class ID
    "unmanned_planes": 2,  # UAV class ID
}

CONFIGS = [
    {
        "label": "Civilian Plane",
        "input_dir": "/dgxa_home/se22ucse255/se_project/civilian_planes/train_civ",
        "annotated_dir": "/dgxa_home/se22ucse255/se_project/civilian_planes/annotated_train_civ",
        "labels_dir": "/dgxa_home/se22ucse255/se_project/civilian_planes/labels_train_civ",
        "class_id": 0  # Set the correct class ID for civilian planes
    },
    {
        "label": "Military Plane",
        "input_dir": "/dgxa_home/se22ucse255/se_project/military_planes/train_mil",
        "annotated_dir": "/dgxa_home/se22ucse255/se_project/military_planes/annotated_train_mil",
        "labels_dir": "/dgxa_home/se22ucse255/se_project/military_planes/labels_train_mil",
        "class_id": 1  # Set the correct class ID for military planes
    },
    {
        "label": "UAV",
        "input_dir": "/dgxa_home/se22ucse255/se_project/unmanned_planes/train_uav",
        "annotated_dir": "/dgxa_home/se22ucse255/se_project/unmanned_planes/annotated_train_uav",
        "labels_dir": "/dgxa_home/se22ucse255/se_project/unmanned_planes/labels_train_uav",
        "class_id": 2  # Set the correct class ID for UAVs
    }
]

# ---------- Utilities ----------
for cfg in CONFIGS:
    os.makedirs(cfg["annotated_dir"], exist_ok=True)
    os.makedirs(cfg["labels_dir"], exist_ok=True)

def resize_image(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=INTERPOLATION)

def process_and_save(img_path, result, annotated_dir, labels_dir, class_id):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping {img_path.name} (unreadable)")
        return

    # Resize
    img_resized = resize_image(img)
    h, w = img_resized.shape[:2]
    label_path = labels_dir / f"{img_path.stem}.txt"
    lines = []

    # Filter for only airplane detections (class 4 in COCO)
    airplane_boxes = [box for box in result.boxes if int(box.cls[0]) == 4]
    
    if not airplane_boxes:
        print(f"No airplane detected in {img_path.name}")
        # Option: Still save an empty label file
        with open(label_path, 'w') as f:
            pass
        return

    # Annotate and prepare label lines
    for box in airplane_boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Use the provided class_id instead of the detected one
        # This maps all airplane detections to the correct class based on the folder

        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + box_w / 2
        y_center = y1 + box_h / 2

        # Normalize
        x_center /= w
        y_center /= h
        box_w /= w
        box_h /= h

        # Use the class_id passed from the config
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # Draw with the correct class ID
        cv2.rectangle(img_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_resized, str(class_id), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save outputs
    cv2.imwrite(str(annotated_dir / img_path.name), img_resized)
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))

# ---------- Main Pipeline ----------
def process_dataset(cfg):
    input_dir = Path(cfg["input_dir"])
    annotated_dir = Path(cfg["annotated_dir"])
    labels_dir = Path(cfg["labels_dir"])
    class_id = cfg["class_id"]  # Get the class_id from the config

    image_paths = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"[{cfg['label']}] Total images: {len(image_paths)}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i:i + BATCH_SIZE]
            imgs = [resize_image(cv2.imread(str(p))) for p in batch]
            results = model(imgs)

            for img_path, result in zip(batch, results):
                executor.submit(process_and_save, img_path, result, annotated_dir, labels_dir, class_id)

            print(f"Processed batch {i//BATCH_SIZE + 1} / {(len(image_paths) + BATCH_SIZE - 1)//BATCH_SIZE}")

# ---------- Run ----------
for cfg in CONFIGS:
    process_dataset(cfg)

print("✅ Resize, Annotation, YOLO Format Complete.")
