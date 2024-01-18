import torch
from torchvision import models, transforms
import torch.nn as nn

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the EfficientNet model
model = models.efficientnet_v2_s(weights=None)
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load('./EfficientNet_best_model.pth'))
model = model.to('cuda')
model.eval();

# Load the sample image
img_path = './Dataset/DRUSEN/41/025_Drusen.jpg'
img_np = cv2.imread(img_path)  # Load the image as a NumPy array
img = Image.fromarray(img_np)  # Convert the NumPy array to a PIL Image

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformations to the image
img_tensor = transform(img).unsqueeze(0).to('cuda')

# Define the class labels
class_labels = ['NORMAL', 'DRUSEN', 'CNV']

# Get the output of the last convolutional layer and the final fully connected layer
features = model.features(img_tensor)
output = model(img_tensor)
# print(features.shape, output.shape)

# Get the class index with the highest score
class_index = torch.argmax(output, dim=1)
# print(class_index)

# Get the weights of the final fully connected layer for the predicted class
weights = model.classifier[-1].weight[class_index]
# print(weights.shape)

# Resize the weights to the size of the last convolutional layer
weights = weights.view(-1, 1280, 1, 1)
weights = torch.nn.functional.interpolate(weights, size=(features.shape[2], features.shape[3]), mode='bilinear', align_corners=False)
# print(weights.shape)

# Apply the weights to the output of the last convolutional layer
cam = torch.sum(weights * features, dim=1, keepdim=True)
# print(cam.shape)

# Normalize the CAM
cam = torch.nn.functional.relu(cam)
cam = cam / torch.max(cam)

cam = cam.squeeze().cpu().detach().numpy()
img = cv2.imread(img_path)
cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
cam = np.uint8(255 * cam)
cam = cv2.convertScaleAbs(cam)

# Create a custom colormap that assigns blue to low values and red to high values
custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
    custom_colormap[i, 0, 0] = max(0, min(255, i - 127))
    custom_colormap[i, 0, 1] = 0
    custom_colormap[i, 0, 2] = max(0, min(255, 127 - i))

# heatmap = cv2.applyColorMap(cam, custom_colormap)
heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_RAINBOW )

overlay = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

# Display the original image and the overlay side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
ax1.imshow(img)
ax1.set_title(f'Class: DRUSEN | Prediction: {class_labels[class_index]}')
ax1.axis('off')

ax2.imshow(overlay)
ax2.set_title(f'Class Activation Map for {class_labels[class_index]}')
ax2.axis('off')

plt.show()