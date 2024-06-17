# feature_extraction.py
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segmentation import UNET, DEVICE

# Paths to the dataset
NORMAL_DIR = '/content/drive/MyDrive/preprocessed_images/normal'
GLAUCOMA_DIR = '/content/drive/MyDrive/preprocessed_images/glaucoma'

# Load the trained UNET model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load('/content/drive/MyDrive/models/modified_unet.pth'))
model.eval()

def segment_image(image, model):
    transform = A.Compose([
        A.Resize(height=160, width=240),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    augmented = transform(image=image)
    image = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = torch.sigmoid(model(image))
        prediction = (prediction > 0.5).float()
    return prediction.squeeze().cpu().numpy()

def load_annotations(directory, model):
    annotations = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            segmented_image = segment_image(image, model)
            annotations[filename] = segmented_image.tolist()
    return annotations

normal_annotations = load_annotations(NORMAL_DIR, model)
glaucoma_annotations = load_annotations(GLAUCOMA_DIR, model)

with open('/content/drive/MyDrive/annotations/normal_annotations.json', 'w') as f:
    json.dump(normal_annotations, f)

with open('/content/drive/MyDrive/annotations/glaucoma_annotations.json', 'w') as f:
    json.dump(glaucoma_annotations, f)