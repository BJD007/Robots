from pymavlink import mavutil
import time
import config

class Autopilot:
    def __init__(self):
        self.vehicle = None

    def connect(self):
        print("Connecting to vehicle...")
        self.vehicle = mavutil.mavlink_connection(config.DRONE_CONNECTION_STRING)
        self.vehicle.wait_heartbeat()
        print("Vehicle connected!")

    def arm(self):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

    def takeoff(self, altitude):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude)

    def land(self):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)

    def goto_position(self, x, y, z):
        self.vehicle.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10, self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000111111111000, x, y, z,
            0, 0, 0, 0, 0, 0, 0, 0))

    def update_position(self, position):
        # Update the drone's position estimate
        pass

    def update_velocity(self, velocity):
        # Update the drone's velocity estimate
        pass
