import random

class OlfactorySystem:
    def __init__(self):
        self.scent_database = {
            "fear": ["sweat", "adrenaline"],
            "happiness": ["serotonin", "dopamine"],
            "stress": ["cortisol", "norepinephrine"]
        }

    def detect_scent(self):
        # Simulating scent detection
        detected_scents = random.choice(list(self.scent_database.values()))
        return detected_scents

    def predict_future_event(self):
        scents = self.detect_scent()
        for emotion, associated_scents in self.scent_database.items():
            if set(scents) == set(associated_scents):
                return f"Predicted future emotion: {emotion}"
        return "Unable to predict future event"
