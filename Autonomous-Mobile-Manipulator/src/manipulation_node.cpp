/*
2.2 Manipulation Algorithms
Using MoveIt! for Inverse Kinematics
Setup: Ensure you have a MoveIt! configuration for your robot. This involves setting up a MoveIt! package using the MoveIt! Setup Assistant, which will generate the necessary configuration files for your robot's URDF model.

File: src/manipulation/src/manipulation_node.cpp

Explanation

    Initialization: The node initializes ROS and MoveIt! interfaces. An AsyncSpinner is used to handle callbacks concurrently.
    Move Group Interface: The MoveGroupInterface is used to interact with the robot's arm. You need to specify the planning group name, which is defined in your MoveIt! configuration.
    Target Pose: A target pose is defined in the world frame. Adjust the position and orientation to suit your task.
    Planning and Execution: The node plans a trajectory to the target pose and executes it if successful. If planning fails, a warning is logged.
*/

#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/PlanningScene.h>
#include <geometry_msgs/Pose.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "manipulation_node");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Initialize MoveIt! interfaces
    moveit::planning_interface::MoveGroupInterface move_group("arm");
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Set target pose
    geometry_msgs::Pose target_pose;
    target_pose.orientation.w = 1.0;
    target_pose.position.x = 0.28;
    target_pose.position.y = -0.2;
    target_pose.position.z = 0.5;
    move_group.setPoseTarget(target_pose);

    // Plan to the target pose
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success) {
        ROS_INFO("Plan successful, executing...");
        move_group.move();
    } else {
        ROS_WARN("Planning failed!");
    }

    ros::shutdown();
    return 0;
}

