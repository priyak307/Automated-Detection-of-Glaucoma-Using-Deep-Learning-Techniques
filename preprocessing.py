# preprocessing.py
from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths to the dataset
NORMAL_DIR = '/content/drive/MyDrive/G1020/Images/normal/'
GLAUCOMA_DIR = '/content/drive/MyDrive/G1020/Images/glaucoma/'
MASK_DIR = '/content/drive/MyDrive/G1020/mask/'

def load_images(directory):
    images = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            images.append(cv2.imread(img_path))
    return images

def load_mask_images(directory):
    masks = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            masks.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    return masks

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

# Load normal and glaucomatous images
normal_images = load_images(NORMAL_DIR)
glaucoma_images = load_images(GLAUCOMA_DIR)

# Load mask images
mask_images = load_mask_images(MASK_DIR)

# Apply CLAHE to all images and masks
processed_images = [apply_clahe(img) for img in normal_images + glaucoma_images]
processed_masks = [apply_clahe(mask) for mask in mask_images]

# Create directories to save the preprocessed images and masks
os.makedirs('/content/drive/MyDrive/preprocessed_images/normal', exist_ok=True)
os.makedirs('/content/drive/MyDrive/preprocessed_images/glaucoma', exist_ok=True)
os.makedirs('/content/drive/MyDrive/preprocessed_masks', exist_ok=True)

# Save the preprocessed images
for i, img in enumerate(processed_images[:len(normal_images)]):
    cv2.imwrite(f'/content/drive/MyDrive/preprocessed_images/normal/image_{i}.jpg', img)
    
for i, img in enumerate(processed_images[len(normal_images):]):
    cv2.imwrite(f'/content/drive/MyDrive/preprocessed_images/glaucoma/image_{i}.jpg', img)

# Save the preprocessed masks
for i, mask in enumerate(processed_masks):
    cv2.imwrite(f'/content/drive/MyDrive/preprocessed_masks/mask_{i}.jpg', mask)