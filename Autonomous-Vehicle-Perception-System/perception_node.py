import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
import cv2
import torch
from torchvision import transforms
from cv_bridge import CvBridge
import numpy as np
from modelselectionNtraining import SimpleCNN, class_mapping

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Detection2DArray, 'object_detections', 10)
        self.bridge = CvBridge()
        num_classes = len(class_mapping)
        state_dict = torch.load('quantized_simple_cnn.pth', map_location=torch.device('cpu'), weights_only=True)
        print("State dict keys:", state_dict.keys())
        self.model = SimpleCNN.from_quantized(state_dict, num_classes)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']  # Update with your actual class names

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        input_tensor = self.transform(cv_image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Process the output
        detections = self.process_output(output[0], cv_image.shape[:2])
        
        # Visualize detections
        vis_image = self.visualize_detections(cv_image, detections)
        
        # Publish detections
        self.publish_detections(detections, msg.header)
        
        # Display the image (for debugging purposes)
        cv2.imshow("Detections", vis_image)
        cv2.waitKey(1)

    def process_output(self, output, image_shape):
        probabilities = torch.softmax(output, dim=0)
        class_id = torch.argmax(probabilities).item()
        confidence = probabilities[class_id].item()
        
        # For simplicity, let's assume the model predicts a single object
        # In a real scenario, you'd need to handle multiple detections
        height, width = image_shape
        x, y = width // 2, height // 2  # Assume object is at the center
        w, h = width // 4, height // 4  # Assume object size is 1/4 of the image

        return [{
            'class_id': class_id,
            'class_name': self.class_names[class_id],
            'confidence': confidence,
            'bbox': [x - w//2, y - h//2, w, h]
        }]

    def visualize_detections(self, image, detections):
        vis_image = image.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return vis_image

    def publish_detections(self, detections, header):
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()
            detection.bbox.center.x = float(det['bbox'][0] + det['bbox'][2] / 2)
            detection.bbox.center.y = float(det['bbox'][1] + det['bbox'][3] / 2)
            detection.bbox.size_x = float(det['bbox'][2])
            detection.bbox.size_y = float(det['bbox'][3])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(det['class_id'])
            hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        self.publisher.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
