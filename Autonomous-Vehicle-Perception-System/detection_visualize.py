import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

class DetectionVisualizer(Node):
    def __init__(self):
        super().__init__('detection_visualizer')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10)
        self.publisher = self.create_publisher(MarkerArray, '/detection_markers', 10)

    def detection_callback(self, msg):
        marker_array = MarkerArray()
        for i, detection in enumerate(msg.detections):
            marker = Marker()
            marker.header = msg.header
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = detection.bbox.center.x
            marker.pose.position.y = detection.bbox.center.y
            marker.pose.position.z = 0.0
            marker.scale.x = detection.bbox.size_x
            marker.scale.y = detection.bbox.size_y
            marker.scale.z = 0.1  # Arbitrary height
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
