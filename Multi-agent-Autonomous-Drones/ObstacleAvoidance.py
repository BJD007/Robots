#Attractive Force: Pulls the drone towards the goal.
#Repulsive Force: Pushes the drone away from obstacles.
#Total Force Calculation: The total force determines the direction and speed of the drone's movement.


class ObstacleAvoidance:
    def __init__(self, drone_position, goal_position, obstacles, repulsive_gain=0.5, attractive_gain=1.0):
        self.drone_position = drone_position
        self.goal_position = goal_position
        self.obstacles = obstacles
        self.repulsive_gain = repulsive_gain
        self.attractive_gain = attractive_gain

    def calculate_forces(self):
        attractive_force = self.attractive_gain * (self.goal_position - self.drone_position)
        repulsive_force = np.zeros(2)

        for obstacle in self.obstacles:
            direction = self.drone_position - obstacle.position
            distance = np.linalg.norm(direction)
            if distance < obstacle.size:
                repulsive_force += self.repulsive_gain * (1 / distance - 1 / obstacle.size) * (direction / distance**2)

        total_force = attractive_force + repulsive_force
        return total_force

    def update_position(self):
        force = self.calculate_forces()
        new_position = self.drone_position + force
        return new_position

# Sample usage
drone_position = np.array([0, 0])
goal_position = np.array([10, 10])
obstacles = [Obstacle(np.array([5, 5]), np.array([1, 1])), Obstacle(np.array([3, 3]), np.array([0.5, 0.5]))]

obstacle_avoidance = ObstacleAvoidance(drone_position, goal_position, obstacles)
for _ in range(50):
    drone_position = obstacle_avoidance.update_position()
    print(f"Updated Position: {drone_position}")
