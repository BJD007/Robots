
#RRT Algorithm: The RRT class is initialized with the start and goal positions, a list of obstacles, and parameters such as maximum iterations and step size.
#Random Node Generation: The get_random_node method generates random nodes, occasionally biasing toward the goal to ensure convergence.
#Nearest Node Calculation: Using KDTree, the nearest node in the existing tree is found for the random node.
#Steering: The steer method calculates the direction from the nearest node to the random node and steps towards it, creating a new node.
#Collision Checking: The check_collision method verifies if the new node collides with any obstacles.
#Path Generation: Once the goal is reached, generate_path backtracks to construct the path.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class RRT:
    def __init__(self, start, goal, obstacles, max_iters=1000, step_size=0.5, goal_sample_rate=0.1):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.max_iters = max_iters
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.nodes = [self.start]

    def plan(self):
        for i in range(self.max_iters):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)
            if not self.check_collision(new_node):
                self.nodes.append(new_node)
                if self.is_goal(new_node):
                    return self.generate_path(new_node)
        return None

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            return Node(np.random.uniform(-10, 10, 2))
        return self.goal

    def get_nearest_node(self, rnd_node):
        tree = KDTree([node.position for node in self.nodes])
        _, idx = tree.query(rnd_node.position)
        return self.nodes[idx]

    def steer(self, from_node, to_node):
        direction = to_node.position - from_node.position
        length = np.linalg.norm(direction)
        direction = direction / length
        new_position = from_node.position + self.step_size * direction
        return Node(new_position, from_node)

    def check_collision(self, node):
        for obstacle in self.obstacles:
            if obstacle.contains(node.position):
                return True
        return False

    def is_goal(self, node):
        return np.linalg.norm(node.position - self.goal.position) < self.step_size

    def generate_path(self, node):
        path = [node.position]
        while node.parent is not None:
            node = node.parent
            path.append(node.position)
        return path[::-1]

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

class Obstacle:
    def __init__(self, position, size):
        self.position = position
        self.size = size

    def contains(self, point):
        return np.all(np.abs(point - self.position) <= self.size)

# Sample usage
start = np.array([0, 0])
goal = np.array([8, 8])
obstacles = [Obstacle(np.array([5, 5]), np.array([1, 1])), Obstacle(np.array([2, 3]), np.array([0.5, 0.5]))]

rrt = RRT(start, goal, obstacles)
path = rrt.plan()

# Visualization
if path is not None:
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], '-r')
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], color='blue')
    for obs in obstacles:
        rect = plt.Rectangle(obs.position - obs.size, obs.size[0]*2, obs.size[1]*2, color='gray')
        plt.gca().add_patch(rect)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.show()
else:
    print("Path not found!")
