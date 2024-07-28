import numpy as np

class EEGSystem:
    def __init__(self):
        self.wave_types = ["alpha", "beta", "theta", "delta"]

    def generate_brain_waves(self):
        # Simulating brain wave data
        return {wave: np.random.random() for wave in self.wave_types}

    def interpret_brain_waves(self, waves):
        dominant_wave = max(waves, key=waves.get)
        if dominant_wave == "alpha":
            return "Relaxed"
        elif dominant_wave == "beta":
            return "Alert"
        elif dominant_wave == "theta":
            return "Drowsy"
        else:
            return "Deep sleep"
