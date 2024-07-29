import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.quantization import DeQuantStub, QuantStub
from kittidataset import KITTIDataset, class_mapping

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    @staticmethod
    def from_quantized(state_dict, num_classes):
        model = SimpleCNN(num_classes)
        
        # Convert quantized fc layers to regular nn.Linear
        fc1_weight = state_dict['fc1._packed_params._packed_params'][0].dequantize()
        fc1_bias = state_dict['fc1._packed_params._packed_params'][1]
        fc2_weight = state_dict['fc2._packed_params._packed_params'][0].dequantize()
        fc2_bias = state_dict['fc2._packed_params._packed_params'][1]
        
        model.fc1 = nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0])
        model.fc2 = nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0])
        
        model.fc1.weight.data.copy_(fc1_weight)
        model.fc1.bias.data.copy_(fc1_bias)
        model.fc2.weight.data.copy_(fc2_weight)
        model.fc2.bias.data.copy_(fc2_bias)
        
        # Load the convolutional layers
        model.conv1.weight.data.copy_(state_dict['conv1.weight'])
        model.conv1.bias.data.copy_(state_dict['conv1.bias'])
        model.conv2.weight.data.copy_(state_dict['conv2.weight'])
        model.conv2.bias.data.copy_(state_dict['conv2.bias'])
        
        return model

    def dequantize(self):
        # This method is no longer needed as we're already using non-quantized layers
        return self

def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, labels

def train_model():
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

if __name__ == '__main__':
    train_model()
