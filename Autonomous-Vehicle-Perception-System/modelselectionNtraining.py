import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from kittidataset import KITTIDataset, class_mapping

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, labels

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

# Initialize model, loss function, and optimizer
num_classes = len(class_mapping)
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        batch_loss = 0
        for i, labels in enumerate(target):
            if len(labels) > 0:
                img_output = output[i].unsqueeze(0)
                img_target = labels.to(device)
                loss = criterion(img_output.expand(len(img_target), -1), img_target)
                batch_loss += loss
        
        batch_loss /= len(target)
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'kitti_cnn.pth')