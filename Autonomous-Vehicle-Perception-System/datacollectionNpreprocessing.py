import os
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

def preprocess_dataset(image_dir, label_dir, output_image_dir, output_label_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))
            
            if os.path.exists(label_path):
                processed_image = preprocess_image(image_path)
                output_image_path = os.path.join(output_image_dir, filename)
                cv2.imwrite(output_image_path, processed_image * 255)
                
                output_label_path = os.path.join(output_label_dir, filename.replace(".png", ".txt"))
                os.system(f'cp {label_path} {output_label_path}')

image_dir = '/home/bhaskarhertzwell/datasets/kitti/training/image_2'
label_dir = '/home/bhaskarhertzwell/datasets/kitti/training/label_2'
output_image_dir = '/home/bhaskarhertzwell/datasets/kitti/preprocessed_images'
output_label_dir = '/home/bhaskarhertzwell/datasets/kitti/preprocessed_labels'
preprocess_dataset(image_dir, label_dir, output_image_dir, output_label_dir)