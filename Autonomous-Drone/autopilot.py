from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import config

class Autopilot:
    def __init__(self):
        self.vehicle = None

    def connect(self):
        print("Connecting to vehicle...")
        self.vehicle = connect(config.DRONE_CONNECTION_STRING, wait_ready=True, baud=57600)
        print("Vehicle connected!")

    def arm_and_takeoff(self, target_altitude):
        print("Basic pre-arm checks")
        while not self.vehicle.is_armable:
            print(" Waiting for vehicle to initialise...")
            time.sleep(1)

        print("Arming motors")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)

        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            print(f" Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def goto_position(self, lat, lon, alt):
        point = LocationGlobalRelative(lat, lon, alt)
        self.vehicle.simple_goto(point)

    def return_to_home(self):
        print("Returning to Home")
        self.vehicle.mode = VehicleMode("RTL")

    def land(self):
        print("Landing")
        self.vehicle.mode = VehicleMode("LAND")

    def get_battery_status(self):
        return self.vehicle.battery.level

    def get_gps_info(self):
        return {
            'lat': self.vehicle.location.global_frame.lat,
            'lon': self.vehicle.location.global_frame.lon,
            'alt': self.vehicle.location.global_frame.alt
        }

    def close_connection(self):
        print("Close vehicle object")
        self.vehicle.close()
