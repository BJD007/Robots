/*

Dependencies
Ensure you have the following dependencies installed:

    ROS
    OpenCV
    PCL
    cv_bridge (for converting ROS images to OpenCV format)
    OpenCV DNN module (for YOLO)

File: src/perception/src/perception_node.cpp


Explanation

    YOLO Object Detection:
        The YOLO model is loaded using OpenCV's DNN module.
        The detectObjects function processes the image to detect objects, drawing bounding boxes around detected objects with a confidence score above a threshold.
    Image Callback:
        The imageCallback function converts the ROS image message to an OpenCV image using cv_bridge.
        It then calls detectObjects to perform object detection and displays the result using OpenCV's GUI functions.
    Point Cloud Processing:
        The pointCloudCallback function processes incoming point cloud data.
        It uses PCL's Voxel Grid filter to downsample the point cloud and a PassThrough filter to limit the Z-axis range, which can help focus on relevant parts of the environment.
    ROS Node Setup:
        The main function initializes the ROS node, sets up subscribers for image and point cloud topics, and enters a spin loop to process incoming data.

This code provides a comprehensive starting point for perception tasks using OpenCV and PCL in a ROS environment. Adjust the parameters and processing steps based on your specific application needs and sensor configurations.
*/



#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

// Load YOLO model
cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");

std::vector<std::string> getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void detectObjects(cv::Mat& frame) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    float confThreshold = 0.5;
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 3);
                std::string label = cv::format("Confidence: %.2f", confidence);
                cv::putText(frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        detectObjects(img);
        cv::imshow("Detected Objects", img);
        cv::waitKey(1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*filtered_cloud);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(filtered_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.5);
    pass.filter(*filtered_cloud);

    // Further processing of filtered_cloud can be done here
    ROS_INFO("Processed point cloud with %lu points", filtered_cloud->points.size());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "perception_node");
    ros::NodeHandle nh;
    ros::Subscriber img_sub = nh.subscribe("camera/image", 1000, imageCallback);
    ros::Subscriber pcl_sub = nh.subscribe("lidar/points", 1000, pointCloudCallback);
    ros::spin();
    return 0;
}
