import time
from sensor_utils import initialize_sensors, read_all_sensors
from drone_control import initialize_drone, check_critical_conditions
from object_detection import detect_falling_objects
from data_transmission import send_data_over_4g
from autopilot import Autopilot
import config

def main():
    # Initialize sensors and drone
    sensors = initialize_sensors()
    autopilot = initialize_drone()

    # Arm and takeoff
    autopilot.arm_and_takeoff(config.TARGET_ALTITUDE)

    while True:
        # Read sensor data
        sensor_data = read_all_sensors(sensors, autopilot)
        
        # Detect falling objects
        falling_object = detect_falling_objects(sensors['camera'])
        sensor_data['falling_object'] = falling_object

        # Send data over 4G
        send_data_over_4g(sensor_data)

        # Check for critical conditions
        check_critical_conditions(autopilot, sensor_data)

        time.sleep(config.LOOP_DELAY)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup code here (close connections, release resources)
        autopilot.close_connection()
