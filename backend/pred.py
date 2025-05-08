import sys
import torch
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Check if the image path is provided
if len(sys.argv) < 2:
    print("Error: No image path provided.")
    sys.exit(1)  # Exit the program if no argument is provided

img_path = sys.argv[1]  # Accept dynamic image path from Node.js

# Model path with .pth extension
model_path = r"c:\Users\laksh\OneDrive - Mahindra University\Desktop\resnet_path\best_resnet_model.pth"  # Update this to your model path

# Define EfficientNet-B0 model using torchvision (same as in training script)
model = efficientnet_b0(pretrained=False)

# Replace final classifier with dropout + final layer (matching the training code)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 3)  # Adjust the number of classes here (3 in this case: civilian, military, uav)
)

# Load the model weights (state_dict)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the model weights onto CPU
model.eval()  # Set the model to evaluation mode

# Image Preprocessing for EfficientNet-B0
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Consistent with training preprocessing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
try:
    img = Image.open(img_path)
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted_class = torch.max(output, 1)

# Define class labels (as per your dataset)
class_labels = ['civilian', 'military', 'uav']  # Replace with your actual class names

# Get the predicted class label and confidence
prediction_label = class_labels[predicted_class.item()]
confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class.item()]

# Print the prediction and confidence
print(f"{prediction_label} ({confidence:.2f})")

# Optional: Visualize the class distribution (if applicable)
label_counts = Counter([prediction_label])
plt.figure(figsize=(8, 5))
plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
plt.title("Prediction Results")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("chart.png")  # Optional
# plt.show()  # Uncomment if you want to display the plot