import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class_mapping = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
    # Add other classes as needed
}

class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.png', '.txt'))
        
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Read label file
        with open(label_name, 'r') as file:
            lines = file.readlines()
            labels = [line.strip().split() for line in lines]
        
        # Convert class names to integer labels
        labels = [class_mapping[label[0]] for label in labels if label[0] in class_mapping]
        
        return image, torch.tensor(labels, dtype=torch.long)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = KITTIDataset('/home/bhaskarhertzwell/datasets/kitti/preprocessed_images',
                             '/home/bhaskarhertzwell/datasets/kitti/preprocessed_labels',
                             transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)