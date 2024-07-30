#This code performs the following tasks:
#Imports necessary libraries for image processing, data splitting, and augmentation.
#Defines a function to preprocess individual images, including resizing and optional augmentation.
#Defines a function to preprocess the entire dataset, including:
#Collecting image and label paths
#Splitting the dataset into train, validation, and test sets
#Processing each image in each split (applying augmentation to training set only)
#Saving processed images and copying corresponding label files
#Sets input and output directory paths
#Calls the preprocessing function to process the entire dataset

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

def preprocess_image(image_path, augment=False):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to 256x256
    image = cv2.resize(image, (256, 256))
    
    if augment:
        # Apply data augmentation if augment is True
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),  # 50% chance of flipping horizontally
            A.RandomBrightnessContrast(p=0.2),  # 20% chance of adjusting brightness/contrast
            A.RandomRotate90(p=0.2),  # 20% chance of rotating 90 degrees
        ])
        image = aug(image=image)['image']
    
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    return image

def preprocess_dataset(image_dir, label_dir, output_dir, test_size=0.2, val_size=0.2):
    images = []
    labels = []
    
    # Collect all image and label paths
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))
            
            if os.path.exists(label_path):
                images.append(image_path)
                labels.append(label_path)
    
    # Split dataset into train, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_size, random_state=42)
    
    splits = [('train', train_images, train_labels), 
              ('val', val_images, val_labels), 
              ('test', test_images, test_labels)]
    
    # Process each split (train, validation, test)
    for split_name, split_images, split_labels in splits:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
        
        for img_path, label_path in zip(split_images, split_labels):
            # Preprocess image (apply augmentation only to training set)
            processed_image = preprocess_image(img_path, augment=(split_name == 'train'))
            output_image_path = os.path.join(split_dir, 'images', os.path.basename(img_path))
            # Save processed image
            cv2.imwrite(output_image_path, (processed_image * 255).astype(np.uint8))
            
            # Copy label file to output directory
            output_label_path = os.path.join(split_dir, 'labels', os.path.basename(label_path))
            os.system(f'cp {label_path} {output_label_path}')

# Set paths for input and output directories
image_dir = '/home/bhaskarhertzwell/datasets/kitti/training/image_2'
label_dir = '/home/bhaskarhertzwell/datasets/kitti/training/label_2'
output_dir = '/home/bhaskarhertzwell/datasets/kitti/preprocessed'

# Run the preprocessing function
preprocess_dataset(image_dir, label_dir, output_dir)
