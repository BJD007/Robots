import serial
import threading
import config

class RFIDDetector:
    def __init__(self):
        self.serial_port = serial.Serial(config.RFID_PORT, config.RFID_BAUD_RATE)
        self.running = False
        self.last_rfid = None

    def start(self):
        self.running = True
        threading.Thread(target=self._detect_rfids).start()

    def _detect_rfids(self):
        while self.running:
            data = self.serial_port.readline().decode().strip()
            if data:
                self.last_rfid = data

    def detect(self):
        return self.last_rfid

    def stop(self):
        self.running = False
        self.serial_port.close()
