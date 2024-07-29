import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.bridge = CvBridge()
        self.image_dir = '/home/bhaskarhertzwell/datasets/kitti/testing/image_2'
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.current_image_index = 0

    def timer_callback(self):
        if self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
            img = cv2.imread(image_path)
            if img is not None:
                msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
                self.publisher_.publish(msg)
                self.get_logger().info(f'Publishing image: {self.image_files[self.current_image_index]}')
                self.current_image_index += 1
            else:
                self.get_logger().warn(f'Failed to read image: {image_path}')
        else:
            self.get_logger().info('All images published. Restarting from the beginning.')
            self.current_image_index = 0

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
