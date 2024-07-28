from naoqi import ALProxy
import time
import random

class TherapySession:
    def __init__(self, ip, port):
        self.motion = ALProxy("ALMotion", ip, port)
        self.tts = ALProxy("ALTextToSpeech", ip, port)
        self.behavior = ALProxy("ALBehaviorManager", ip, port)
        self.face_detection = ALProxy("ALFaceDetection", ip, port)
        self.memory = ALProxy("ALMemory", ip, port)
        self.audio_player = ALProxy("ALAudioPlayer", ip, port)

    def start_session(self):
        self.tts.say("Hello! Let's start our therapy session.")
        time.sleep(1)
        self.turn_taking_game()
        self.eye_contact_exercise()
        self.imitation_game()
        self.tts.say("Great job! Our session is complete. See you next time!")

    def turn_taking_game(self):
        self.tts.say("Let's play a turn-taking game. I'll say a color, then you say a color. Ready?")
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for i in range(3):
            nao_color = random.choice(colors)
            self.tts.say(f"My turn. {nao_color}")
            time.sleep(3)
            self.tts.say("Your turn. What color do you choose?")
            time.sleep(5)  # Wait for user response
            self.tts.say("Great! Let's continue.")

    def eye_contact_exercise(self):
        self.tts.say("Now, let's practice making eye contact. Can you look at my eyes?")
        self.face_detection.subscribe("FaceDetection")
        start_time = time.time()
        duration = 30  # Exercise duration in seconds

        while time.time() - start_time < duration:
            face_data = self.memory.getData("FaceDetected")
            if face_data and face_data[1] != []:
                self.tts.say("I see you! Great job maintaining eye contact.")
                time.sleep(2)
            else:
                self.tts.say("Can you look at my eyes?")
                time.sleep(2)

        self.face_detection.unsubscribe("FaceDetection")

    def imitation_game(self):
        self.tts.say("Let's play an imitation game. I'll do an action, and you try to copy me.")
        actions = [
            ("raise_arm", "I'm raising my right arm. Can you do the same?"),
            ("wave", "I'm waving. Can you wave back?"),
            ("nod", "I'm nodding my head. Can you nod your head too?")
        ]

        for action, instruction in actions:
            self.behavior.runBehavior(f"animations/{action}")
            self.tts.say(instruction)
            time.sleep(5)  # Wait for user to imitate
            self.tts.say("Great imitation! Let's try another one.")

if __name__ == "__main__":
    IP = "192.168.1.100"  # Replace with your NAO's IP address
    PORT = 9559

    therapy = TherapySession(IP, PORT)
    therapy.start_session()