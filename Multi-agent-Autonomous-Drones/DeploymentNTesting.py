
#Test Cases: Validates the path planning and obstacle avoidance algorithms.
#MQTT Connection Test: Ensures the MQTT client can connect to the broker successfully.

import unittest
import numpy as np

class TestNavigationSystem(unittest.TestCase):
    def test_rrt_path_planning(self):
        start = np.array([0, 0])
        goal = np.array([8, 8])
        obstacles = [Obstacle(np.array([5, 5]), np.array([1, 1]))]
        rrt = RRT(start, goal, obstacles)
        path = rrt.plan()
        self.assertIsNotNone(path, "Path should not be None")

    def test_obstacle_avoidance(self):
        drone_position = np.array([0, 0])
        goal_position = np.array([10, 10])
        obstacles = [Obstacle(np.array([5, 5]), np.array([1, 1]))]
        obstacle_avoidance = ObstacleAvoidance(drone_position, goal_position, obstacles)
        new_position = obstacle_avoidance.update_position()
        self.assertFalse(np.array_equal(drone_position, new_position), "Drone position should change")

class TestCommunicationSystem(unittest.TestCase):
    def test_mqtt_connection(self):
        communication = DroneCommunication("broker.hivemq.com", "drone1")
        communication.connect()
        self.assertTrue(communication.client.is_connected(), "Client should be connected")
        communication.disconnect()

if __name__ == '__main__':
    unittest.main()
