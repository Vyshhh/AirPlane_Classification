import torch

# Load the model weights from the .pth file
model_path = r"c:\Users\laksh\OneDrive - Mahindra University\Desktop\resnet_path\best_resnet_model.pth" 
model_weights = torch.load(model_path, map_location=torch.device('cpu'))

# Check if the loaded object is a dictionary (which it often is)
if isinstance(model_weights, dict):
    print("Keys in the state_dict:")
    for key in model_weights.keys():
        print(key)  # Print all keys in the model weights
else:
    print("The loaded file is not a dictionary, it might be the full model.")
