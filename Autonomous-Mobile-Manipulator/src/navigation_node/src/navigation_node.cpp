/* 2.3 Navigation Algorithms
Using ROS Navigation Stack
Setup: Ensure you have a map and robot configuration for navigation. This includes setting up the move_base node with appropriate configurations for your robot's sensors and actuators. File: src/navigation/src/navigation_node.cpp

Explanation

    Action Client: The MoveBaseClient is set up to communicate with the move_base action server, which handles path planning and execution.
    Goal Definition: A navigation goal is defined with a specific position and orientation in the map frame. Adjust these values to navigate to different locations.
    Goal Execution: The goal is sent to the move_base server, and the client waits for the result. The success or failure of the movement is logged.
    ROS Spin: The node uses a simple loop to wait for the action server and process results.

These implementations provide a robust framework for robotic manipulation and navigation tasks. Customize the parameters and configurations to suit your robot's specific setup and operational environment.

*/

#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv) {
    ros::init(argc, argv, "navigation_node");
    MoveBaseClient ac("move_base", true);

    // Wait for the action server to start
    while (!ac.waitForServer(ros::Duration(5.0))) {
        ROS_INFO("Waiting for the move_base action server to come up");
    }

    // Define a goal to send to the move_base
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.pose.position.x = 1.0;
    goal.target_pose.pose.position.y = 1.0;
    goal.target_pose.pose.orientation.w = 1.0;

    // Send the goal
    ROS_INFO("Sending goal");
    ac.sendGoal(goal);

    // Wait for the result
    ac.waitForResult();

    // Check the result
    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("The robot moved to the goal!");
    } else {
        ROS_WARN("The robot failed to move to the goal.");
    }

    return 0;
}
