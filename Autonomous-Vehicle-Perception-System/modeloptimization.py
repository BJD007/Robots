import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from modelselectionNtraining import SimpleCNN, class_mapping

# Get the number of classes from your class_mapping
num_classes = len(class_mapping)

# Load the trained model
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('kitti_cnn.pth'))

# Apply dynamic quantization
quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model.state_dict(), 'quantized_simple_cnn.pth')
