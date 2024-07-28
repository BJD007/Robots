This code creates a robot-assisted therapy session using the AIsoy robot, focusing on the three main objectives you mentioned. 

Here's a breakdown of the code:
We import the necessary modules and initialize the AIsoy robot.
We define three main activities: eye contact, turn-taking, and imitation.
The therapy_session() function serves as the main controller, randomly selecting and running 5 activities.
eye_contact_exercise() encourages the child to make eye contact with the robot, providing positive reinforcement.
turn_taking_exercise() practices turn-taking skills using a color-selection game.
imitation_exercise() promotes verbal and motor imitation by asking the child to perform actions and then demonstrating them.

Each exercise includes verbal instructions, wait times for the child to respond, and positive reinforcement. The robot also uses emotions and LED colors to enhance engagement.
Note that this code assumes the existence of certain methods in the AIsoy robot API (like setEmotion, moveHead, setLedColor, etc.). You may need to adjust these method names or parameters based on the actual AIsoy API documentation.
Additionally, you should consider adding error handling, more varied activities, and customization options based on individual children's needs and preferences. Always ensure that a qualified therapist oversees the use of this robot-assisted therapy program.