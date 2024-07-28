import cv2
from pyzbar import pyzbar
import threading

class BarcodeDetector:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.running = False
        self.last_barcode = None

    def start(self):
        self.running = True
        threading.Thread(target=self._detect_barcodes).start()

    def _detect_barcodes(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                barcodes = pyzbar.decode(frame)
                for barcode in barcodes:
                    self.last_barcode = barcode.data.decode('utf-8')

    def detect(self):
        return self.last_barcode

    def stop(self):
        self.running = False
        self.camera.release()
