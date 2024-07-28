import aisoy
import time
import random

# Initialize the AIsoy robot
robot = aisoy.Robot()

# Define therapy activities
activities = [
    "eye_contact",
    "turn_taking",
    "imitation"
]

# Main therapy session function
def therapy_session():
    robot.speak("Hello! Let's start our therapy session.")
    robot.setEmotion("happy")

    for _ in range(5):  # Perform 5 random activities
        activity = random.choice(activities)
        if activity == "eye_contact":
            eye_contact_exercise()
        elif activity == "turn_taking":
            turn_taking_exercise()
        elif activity == "imitation":
            imitation_exercise()

    robot.speak("Great job! Our session is complete.")
    robot.setEmotion("proud")

# Eye contact exercise
def eye_contact_exercise():
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

# Turn-taking exercise
def turn_taking_exercise():
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

# Imitation exercise
def imitation_exercise():
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

# Run the therapy session
therapy_session()