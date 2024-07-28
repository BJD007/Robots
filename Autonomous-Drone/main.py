import time
from autopilot import Autopilot
from video_stream import VideoStream
from user_app import UserApp
from follow_me import FollowMe
from obstacle_detection import ObstacleDetection
from charging import AutonomousCharging
from mapping import LiveMapping
import config

def main():
    # Initialize systems
    autopilot = Autopilot()
    video_stream = VideoStream()
    user_app = UserApp()
    follow_me = FollowMe()
    obstacle_detection = ObstacleDetection()
    charging = AutonomousCharging()
    mapping = LiveMapping()

    # Connect to the drone
    autopilot.connect()

    # Arm and takeoff
    autopilot.arm_and_takeoff(config.TARGET_ALTITUDE)

    # Start video streaming
    video_stream.start_stream()

    # Start user application
    user_app.start()

    # Start follow-me mode
    follow_me.start(autopilot)

    # Start obstacle detection
    obstacle_detection.start(autopilot)

    # Start live mapping
    mapping.start(autopilot)

    # Main loop
    while True:
        # Check for user commands
        user_app.check_commands(autopilot)

        # Check for obstacle detection
        obstacle_detection.check_obstacles()

        # Check battery level for autonomous charging
        if autopilot.get_battery_status() < config.BATTERY_THRESHOLD:
            charging.start_charging(autopilot)

        time.sleep(config.LOOP_DELAY)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup code here (close connections, release resources)
        pass
