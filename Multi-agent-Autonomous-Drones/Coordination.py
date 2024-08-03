
#Task and Drone Classes: Represent tasks with positions and drones with capabilities.
#Bid Calculation: Each drone calculates a bid for a task based on distance and speed.
#Task Assignment: The drone with the lowest bid is assigned the task.


class TaskAllocation:
    def __init__(self, drones, tasks):
        self.drones = drones
        self.tasks = tasks

    def allocate_tasks(self):
        for task in self.tasks:
            best_drone = None
            best_bid = float('inf')
            for drone in self.drones:
                bid = drone.calculate_bid(task)
                if bid < best_bid:
                    best_bid = bid
                    best_drone = drone
            if best_drone is not None:
                best_drone.assign_task(task)

class Drone:
    def __init__(self, id, capabilities):
        self.id = id
        self.capabilities = capabilities
        self.current_task = None

    def calculate_bid(self, task):
        # Example: calculate bid based on distance and capabilities
        distance = np.linalg.norm(self.position - task.position)
        return distance / self.capabilities['speed']

    def assign_task(self, task):
        self.current_task = task
        print(f"Drone {self.id} assigned to task {task.id}")

class Task:
    def __init__(self, id, position):
        self.id = id
        self.position = position

# Sample usage
drones = [Drone(1, {'speed': 2.0}), Drone(2, {'speed': 1.5}), Drone(3, {'speed': 1.8})]
tasks = [Task(1, np.array([4, 4])), Task(2, np.array([8, 8]))]

task_allocation = TaskAllocation(drones, tasks)
task_allocation.allocate_tasks()
