import aisoy
import time
import random

# Initialize the AIsoy robot
robot = aisoy.Robot()

# Define therapy activities
activities = [
    "eye_contact",
    "turn_taking",
    "imitation",
    "emotion_recognition",  # Added new activity
    "storytelling"  # Added new activity
]

# Main therapy session function
def therapy_session(duration=15):  # Added customizable duration
    try:
        robot.speak("Hello! Let's start our therapy session.")
        robot.setEmotion("happy")

        start_time = time.time()
        while time.time() - start_time < duration * 60:  # Run for specified minutes
            activity = random.choice(activities)
            if activity == "eye_contact":
                eye_contact_exercise()
            elif activity == "turn_taking":
                turn_taking_exercise()
            elif activity == "imitation":
                imitation_exercise()
            elif activity == "emotion_recognition":
                emotion_recognition_exercise()
            elif activity == "storytelling":
                storytelling_exercise()

        robot.speak("Great job! Our session is complete.")
        robot.setEmotion("proud")
    except Exception as e:
        print(f"An error occurred: {e}")
        robot.speak("I'm sorry, but we need to stop the session due to a problem.")

# Eye contact exercise
def eye_contact_exercise():
    try:
        robot.speak("Let's practice making eye contact.")
        robot.setEmotion("neutral")
        robot.moveHead(0, 0)  # Center head position
        
        for _ in range(3):
            robot.speak("Can you look at my eyes?")
            time.sleep(5)  # Wait for the child to make eye contact
            robot.setEmotion("happy")
            robot.speak("Great job!")
            time.sleep(2)
        
        robot.speak("You did very well with eye contact!")
    except AttributeError:
        print("Error: 'moveHead' method not found. Please check the AIsoy API documentation.")

# Turn-taking exercise
def turn_taking_exercise():
    try:
        robot.speak("Now, let's practice taking turns.")
        colors = ["red", "blue", "green", "yellow"]
        
        for _ in range(4):
            robot_color = random.choice(colors)
            robot.speak(f"My turn. I choose {robot_color}.")
            robot.setLedColor(robot_color)
            time.sleep(2)
            
            robot.speak("Your turn. What color do you choose?")
            time.sleep(5)  # Wait for the child's response
            robot.speak("Great choice!")
        
        robot.speak("You're doing great at taking turns!")
    except AttributeError:
        print("Error: 'setLedColor' method not found. Please check the AIsoy API documentation.")

# Imitation exercise
def imitation_exercise():
    try:
        robot.speak("Let's play a fun imitation game!")
        actions = [
            ("Raise your hand", "raiseArm"),
            ("Wave hello", "wave"),
            ("Nod your head", "nod"),
            ("Shake your head", "shake")
        ]
        
        for action, robot_action in actions:
            robot.speak(f"Can you {action}?")
            time.sleep(5)  # Wait for the child to perform the action
            robot.speak("Now watch me.")
            getattr(robot, robot_action)()
            robot.speak("Great imitation!")
        
        robot.speak("You're excellent at imitating!")
    except AttributeError as e:
        print(f"Error: Method '{e.name}' not found. Please check the AIsoy API documentation.")

# New emotion recognition exercise
def emotion_recognition_exercise():
    try:
        robot.speak("Let's play an emotion guessing game!")
        emotions = ["happy", "sad", "angry", "surprised"]
        
        for _ in range(3):
            emotion = random.choice(emotions)
            robot.setEmotion(emotion)
            robot.speak("Can you guess what emotion I'm showing?")
            time.sleep(7)  # Wait for the child's response
            robot.speak(f"I was feeling {emotion}. Great guess!")
        
        robot.speak("You're very good at recognizing emotions!")
    except AttributeError:
        print("Error: 'setEmotion' method not found. Please check the AIsoy API documentation.")

# New storytelling exercise
def storytelling_exercise():
    try:
        robot.speak("Let's create a story together!")
        story_parts = ["Once upon a time", "there was a brave robot", "who loved to explore", "and make new friends"]
        
        for part in story_parts:
            robot.speak(part)
            robot.speak("What happens next in our story?")
            time.sleep(10)  # Wait for the child's contribution
            robot.speak("That's a great idea! Let's continue.")
        
        robot.speak("The end. What a wonderful story we created together!")
    except Exception as e:
        print(f"An error occurred during storytelling: {e}")

# Run the therapy session
if __name__ == "__main__":
    try:
        # Customization options (could be set based on individual needs)
        session_duration = 20  # minutes
        therapy_session(duration=session_duration)
    except KeyboardInterrupt:
        print("Session interrupted by user.")
        robot.speak("Goodbye! I hope you had fun.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        robot.speak("I'm sorry, but we need to end our session unexpectedly.")
