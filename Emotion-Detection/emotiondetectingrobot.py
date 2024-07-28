class EmotionDetectingRobot:
    def __init__(self):
        self.vision = VisionSystem()
        self.speech = SpeechRecognitionSystem()
        self.olfactory = OlfactorySystem()
        self.eeg = EEGSystem()
        self.emotion_detector = EmotionDetectionSystem(self.vision, self.speech, self.olfactory, self.eeg)

    def run(self):
        while True:
            emotion = self.emotion_detector.detect_emotion()
            print(f"Detected emotion: {emotion}")
            # Here you would add code to make the robot respond based on the detected emotion

if __name__ == "__main__":
    robot = EmotionDetectingRobot()
    robot.run()
