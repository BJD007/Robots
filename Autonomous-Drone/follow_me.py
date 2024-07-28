import time
import config

class FollowMe:
    def __init__(self):
        self.following = False

    def start(self, autopilot):
        self.following = True
        threading.Thread(target=self._follow, args=(autopilot,)).start()

    def _follow(self, autopilot):
        while self.following:
            # Implement follow-me logic
            # Example: Move to the current GPS location of the target
            target_gps = self.get_target_gps()
            autopilot.goto_position(target_gps['lat'], target_gps['lon'], config.TARGET_ALTITUDE)
            time.sleep(config.FOLLOW_ME_DELAY)

    def stop(self):
        self.following = False

    def get_target_gps(self):
        # Implement logic to get the target GPS location
        return {'lat': 0, 'lon': 0}
