/*Explanation

To create a complete behavior tree using the BehaviorTree.CPP library, we need to define a behavior tree structure, implement the nodes, and execute the tree. Below is a detailed implementation, including how to set up the behavior tree and define custom nodes.
Dependencies

    BehaviorTree.CPP Library: Install the BehaviorTree.CPP library. You can find the installation instructions on the official GitHub repository.
    ROS: Ensure your ROS environment is set up.

File: src/behavior_tree/src/behavior_tree_node.cpp


    Custom Nodes:
        MoveToGoal: This is a synchronous action node that simulates moving to a specified goal. It takes a "goal" input port.
        IsBatteryLow: This is a condition node that checks if the battery is low. It returns FAILURE if the battery is low, preventing further actions.
    Behavior Tree Structure:
        The behavior tree is defined in XML format. In this example, a simple sequence is used:
            IsBatteryLow: Checks the battery status.
            MoveToGoal: Executes the move action if the battery is not low.
    Execution:
        The tree is ticked in a loop, simulating continuous operation. Adjust the rate as needed for your application.
    Ports and Configuration:
        Custom nodes define input and output ports using providedPorts(). This allows passing data between nodes in the tree.
    XML Definition:
        The behavior tree is defined as an XML string within the code. Alternatively, you can load it from an external XML file.

This setup provides a flexible framework for defining and executing complex robot behaviors using behavior trees. Customize the nodes and tree structure to match your specific application requirements.

*/


/*

Behavior Tree Node
Ensure your behavior tree node is set up to interact with the simulated environment. This might involve subscribing to sensor topics and publishing commands to actuators.
Testing and Refining the Behavior Tree

    Launch the Simulation:
        Start the simulation using the launch file:

        bash
        roslaunch your_package simulation.launch

Monitor and Adjust:

    Use ROS tools like rqt and rviz to monitor the robot's state and sensor data.
    Observe how the behavior tree executes actions and conditions. Look for any unexpected behaviors or failures.

Modify the Behavior Tree:

    Based on observations, you may need to adjust the behavior tree. This could involve:
        Adding new conditions or actions.
        Adjusting parameters or thresholds.
        Reordering nodes to prioritize different tasks.

Iterate:

    After making changes, restart the simulation and test again.
    Continue this process until the robot behaves as expected in the simulated environment.


Tips for Refinement

    Logging: Use ROS logging to track the execution of each node and identify issues.
    Simulation Speed: Adjust the simulation speed if needed to test different scenarios more quickly.
    Environment Variability: Test the behavior tree under different environmental conditions to ensure robustness.

This process will help you refine your behavior tree to work effectively in a simulated environment, preparing it for real-world deployment.

*/

#include <behaviortree_cpp_v3/bt_factory.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

// Custom Action Node: MoveToGoal
class MoveToGoal : public BT::SyncActionNode {
public:
    MoveToGoal(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts() {
        return { BT::InputPort<std::string>("goal") };
    }

    BT::NodeStatus tick() override {
        std::string goal;
        if (!getInput<std::string>("goal", goal)) {
            throw BT::RuntimeError("missing required input [goal]");
        }
        ROS_INFO("Moving to goal: %s", goal.c_str());
        // Implement logic to send movement commands to the robot
        // Simulate success or failure
        return BT::NodeStatus::SUCCESS;
    }
};

// Custom Condition Node: IsBatteryLow
class IsBatteryLow : public BT::ConditionNode {
public:
    IsBatteryLow(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config) {}

    BT::NodeStatus tick() override {
        // Check battery status from a simulated sensor
        bool battery_low = false; // Replace with actual check
        ROS_INFO("Checking battery status: %s", battery_low ? "Low" : "OK");
        return battery_low ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "behavior_tree_node");
    ros::NodeHandle nh;

    BT::BehaviorTreeFactory factory;
    factory.registerNodeType<MoveToGoal>("MoveToGoal");
    factory.registerNodeType<IsBatteryLow>("IsBatteryLow");

    // Define the behavior tree structure in XML
    const char* xml_text = R"(
    <root main_tree_to_execute="MainTree">
        <BehaviorTree ID="MainTree">
            <Sequence>
                <IsBatteryLow/>
                <MoveToGoal goal="Position1"/>
            </Sequence>
        </BehaviorTree>
    </root>
    )";

    auto tree = factory.createTreeFromText(xml_text);

    ros::Rate rate(1); // 1 Hz
    while (ros::ok()) {
        tree.tickRoot();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
