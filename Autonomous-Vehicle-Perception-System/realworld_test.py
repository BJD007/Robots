import torch
import cv2
import numpy as np
from torchvision import transforms

# Load your quantized model
model = torch.jit.load('quantized_simple_cnn1.pth')
model.eval()

# Set up camera
cap = cv2.VideoCapture(0)  # Use appropriate camera index

# Image processing function
def process_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Main testing loop
def test_model(model):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = process_image(frame)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process output (e.g., object detection, classification)
        print(output)
        
        # Visualize results on frame
        # ... (implement visualization based on your model's output)
        
        cv2.imshow('Real-world Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the test
test_model(model)
