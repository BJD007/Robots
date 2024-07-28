import serial
import threading

class OpticalFlow:
    def __init__(self):
        self.serial_port = serial.Serial(config.PX4FLOW_PORT, config.PX4FLOW_BAUD_RATE)
        self.velocity = [0, 0, 0]
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._read_optical_flow_data).start()

    def _read_optical_flow_data(self):
        while self.running:
            data = self.serial_port.readline().decode().strip()
            # Parse PX4Flow data and update velocity
            # This is a placeholder - implement actual PX4Flow data parsing
            self.velocity = [float(x) for x in data.split(',')]

    def get_velocity(self):
        return self.velocity

    def stop(self):
        self.running = False
