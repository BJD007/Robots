# Drone configuration
DRONE_CONNECTION_STRING = '/dev/ttyACM0'

# Sensor pins
FLAME_SENSOR_PIN = 17
GAS_SENSOR_PIN = 18
SOIL_MOISTURE_PIN = 27
HUMAN_DETECTION_PIN = 22

# 4G configuration
APN = "your_apn_here"
USERNAME = "your_username"
PASSWORD = "your_password"
SERVER_URL = "http://your_server_url.com/api/data"

# Other settings
LOOP_DELAY = 1  # seconds


# Add this to config.py
TARGET_ALTITUDE = 10  # meters