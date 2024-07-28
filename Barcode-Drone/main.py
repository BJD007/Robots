import time
from autopilot import Autopilot
from uwb_positioning import UWBPositioning
from optical_flow import OpticalFlow
from barcode_detector import BarcodeDetector
from mission_planner import MissionPlanner
import config

def main():
    # Initialize systems
    autopilot = Autopilot()
    uwb = UWBPositioning()
    optical_flow = OpticalFlow()
    barcode_detector = BarcodeDetector()
    mission_planner = MissionPlanner()

    # Connect to the drone
    autopilot.connect()

    # Start UWB positioning
    uwb.start()

    # Start optical flow
    optical_flow.start()

    # Start barcode detection
    barcode_detector.start()

    # Plan and execute mission
    mission_planner.plan_mission()
    mission_planner.execute_mission(autopilot, uwb, optical_flow, barcode_detector)

    # Main loop
    while True:
        # Update position
        position = uwb.get_position()
        autopilot.update_position(position)

        # Update velocity
        velocity = optical_flow.get_velocity()
        autopilot.update_velocity(velocity)

        # Check for barcodes
        barcode = barcode_detector.detect()
        if barcode:
            print(f"Detected barcode: {barcode}")

        # Check mission status
        if mission_planner.is_mission_complete():
            break

        time.sleep(config.LOOP_DELAY)

    # Land the drone
    autopilot.land()

if __name__ == "__main__":
    main()
