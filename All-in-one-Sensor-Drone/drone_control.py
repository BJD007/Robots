from autopilot import Autopilot
import config

def initialize_drone():
    autopilot = Autopilot()
    autopilot.connect()
    return autopilot

def check_critical_conditions(autopilot, sensor_data):
    if sensor_data['flame'] or sensor_data['gas']:
        autopilot.return_to_launch()
