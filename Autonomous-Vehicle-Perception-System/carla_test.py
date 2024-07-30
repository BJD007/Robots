import carla
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load your quantized model
model = torch.jit.load('quantized_simple_cnn1.pth')
model.eval()

# Set up CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

# Set up camera sensor
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '110')

# Spawn camera
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform)

# Image processing function
def process_image(image):
    img = np.array(image.raw_data).reshape((720, 1280, 4))[:,:,:3]
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Main testing loop
def test_model(model, camera):
    while True:
        image = camera.listen(lambda image: process_image(image))
        with torch.no_grad():
            output = model(image)
        # Process output (e.g., object detection, classification)
        print(output)

# Run the test
test_model(model, camera)
