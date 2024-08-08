/*
Implementing an object retrieval use case involves creating a ROS node that can interact with the robot's sensors and actuators to locate, approach, and manipulate an object.
File: src/use_cases/src/object_retrieval_node.cpp

Explanation

    Object Detection:
        The detectObject function is a placeholder for object detection logic. You can replace it with OpenCV-based methods or a machine learning model to detect objects in the image.
    Movement Control:
        The moveToTarget function uses simple proportional control to align the robot with the detected object. It publishes velocity commands to the cmd_vel topic to steer the robot.
    Manipulation:
        The manipulateObject function uses MoveIt! to plan and execute a motion to a predefined target pose. This simulates the robot's arm reaching for the object.
    Image Processing:
        The imageCallback function processes incoming camera images, detects objects, and initiates movement towards them.
    ROS Node:
        The node initializes ROS, sets up publishers and subscribers, and enters a spin loop to process incoming data.

Testing

    Simulation:
        Launch your Gazebo simulation with the robot and environment set up.
        Run the object_retrieval_node and observe the robot's behavior as it attempts to locate and retrieve objects.
    Real-World Testing:
        Deploy the node on the physical robot and test in a controlled environment.
        Adjust detection thresholds, movement parameters, and target poses as needed to improve performance.

This implementation provides a basic framework for an object retrieval task. You can expand and refine it with more sophisticated detection and control algorithms to handle various objects and environments.

*/


#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

// Function to detect object using OpenCV
bool detectObject(const cv::Mat& image, cv::Rect& object_rect) {
    // Dummy implementation: Replace with actual object detection logic
    // For example, use color thresholding or a pre-trained model
    object_rect = cv::Rect(100, 100, 50, 50); // Example bounding box
    return true; // Return true if object is detected
}

// Function to move the robot towards the object
void moveToTarget(ros::Publisher& cmd_vel_pub, const cv::Rect& object_rect) {
    geometry_msgs::Twist move_cmd;
    // Simple proportional control to center the object
    if (object_rect.x < 320) {
        move_cmd.angular.z = 0.1; // Turn left
    } else if (object_rect.x > 320) {
        move_cmd.angular.z = -0.1; // Turn right
    } else {
        move_cmd.linear.x = 0.1; // Move forward
    }
    cmd_vel_pub.publish(move_cmd);
}

// Function to manipulate the object using MoveIt!
void manipulateObject() {
    moveit::planning_interface::MoveGroupInterface move_group("arm");

    // Define target pose for the end effector
    geometry_msgs::Pose target_pose;
    target_pose.orientation.w = 1.0;
    target_pose.position.x = 0.5; // Example position
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.5;

    move_group.setPoseTarget(target_pose);

    // Plan and execute the motion
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
        ROS_INFO("Manipulating object...");
        move_group.move();
    } else {
        ROS_WARN("Failed to plan manipulation!");
    }
}

// Callback function for image processing
void imageCallback(const sensor_msgs::ImageConstPtr& msg, ros::Publisher& cmd_vel_pub) {
    cv::Mat img;
    try {
        img = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return;
    }

    cv::Rect object_rect;
    if (detectObject(img, object_rect)) {
        moveToTarget(cmd_vel_pub, object_rect);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_retrieval_node");
    ros::NodeHandle nh;

    // Publisher for robot movement
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);

    // Subscriber for camera image
    ros::Subscriber img_sub = nh.subscribe<sensor_msgs::Image>("camera/image", 10,
        boost::bind(imageCallback, _1, boost::ref(cmd_vel_pub)));

    // Spin to process callbacks
    ros::spin();

    return 0;
}
