import time
from autopilot import Autopilot
from uwb_positioning import UWBPositioning
from optical_flow import OpticalFlow
from rfid_detector import RFIDDetector
from mission_planner import MissionPlanner
import config

def main():
    # Initialize systems
    autopilot = Autopilot()
    uwb = UWBPositioning()
    optical_flow = OpticalFlow()
    rfid_detector = RFIDDetector()
    mission_planner = MissionPlanner()

    # Connect to the drone
    autopilot.connect()

    # Start UWB positioning
    uwb.start()

    # Start optical flow
    optical_flow.start()

    # Start RFID detection
    rfid_detector.start()

    # Plan and execute mission
    mission_planner.plan_mission()
    mission_planner.execute_mission(autopilot, uwb, optical_flow, rfid_detector)

    # Main loop
    while True:
        # Update position
        position = uwb.get_position()
        autopilot.update_position(position)

        # Update velocity
        velocity = optical_flow.get_velocity()
        autopilot.update_velocity(velocity)

        # Check for RFIDs
        rfid = rfid_detector.detect()
        if rfid:
            print(f"Detected RFID: {rfid}")
            mission_planner.handle_rfid_detection(rfid)

        # Check mission status
        if mission_planner.is_mission_complete():
            break

        time.sleep(config.LOOP_DELAY)

    # Land the drone
    autopilot.land()

if __name__ == "__main__":
    main()
