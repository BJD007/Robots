import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, DeQuantStub, QuantStub
from kittidataset import KITTIDataset, class_mapping
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.quant = QuantStub()  # Quantization stub for input
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second convolutional layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling layer
        self.fc1 = nn.Linear(32, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Second fully connected layer
        self.dequant = DeQuantStub()  # Dequantization stub for output

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)  # Apply adaptive pooling
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU activation
        x = self.fc2(x)  # Apply second fully connected layer
        x = self.dequant(x)  # Dequantize output
        return x

# Custom collate function for DataLoader
def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, labels

# Function to calculate accuracy, precision, and recall
def calculate_metrics(predictions, targets):
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall

# Main training function
def train_model(model_type):
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
    model = SimpleCNN(num_classes)  # Change this to get_model(model_type, num_classes) for other models
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        start_time = time.time()  # Start time for inference benchmarking

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Collect predictions and targets for metrics calculation
            all_preds.append(output)
            all_targets.append(target)

        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        accuracy, precision, recall = calculate_metrics(all_preds, all_targets)

        # Calculate inference time
        end_time = time.time()
        inference_time = end_time - start_time

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'Inference Time: {inference_time:.4f} seconds')

    # Save the trained model
    torch.save(model.state_dict(), f'kitti_{model_type}.pth')

    # Apply dynamic quantization
    quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # Save the quantized model
    torch.save(quantized_model.state_dict(), f'quantized_{model_type}.pth')
    print(f'Model quantization completed. Quantized model saved as "quantized_{model_type}.pth"')

# Main execution block
if __name__ == '__main__':
    model_type = 'simplecnn'  # Change this to 'unet', 'vit', 'yolo', or 'ssd' as needed
    train_model(model_type)
