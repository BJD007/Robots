import cv2
import threading

class VideoStream:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.streaming = False

    def start_stream(self):
        self.streaming = True
        threading.Thread(target=self._stream).start()

    def _stream(self):
        while self.streaming:
            ret, frame = self.camera.read()
            if ret:
                # Show live video feed
                cv2.imshow('Live Video Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.camera.release()
        cv2.destroyAllWindows()

    def stop_stream(self):
        self.streaming = False
