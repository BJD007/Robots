import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

class VisionSystem:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.model = MobileNetV2(weights='imagenet')

    def capture_image(self):
        ret, frame = self.camera.read()
        return frame

    def recognize_objects(self, image):
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        return decode_predictions(predictions, top=3)[0]

    def process_frame(self):
        frame = self.capture_image()
        objects = self.recognize_objects(frame)
        return objects
