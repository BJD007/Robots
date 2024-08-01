# Automated-Railway-Crossing-Barrier

This project implements an automated railway crossing barrier using Arduino, stepper motors, and sensors.

## Directory Structure

AutomatedRailwayCrossing/
├── src/
│ ├── main.cpp
│ ├── Barrier.cpp
│ ├── Barrier.h
│ ├── Sensors.cpp
│ ├── Sensors.h
│ ├── Warnings.cpp
│ ├── Warnings.h
│ ├── PowerManagement.cpp
│ ├── PowerManagement.h
├── lib/
│ └── Stepper/
│ ├── Stepper.cpp
│ ├── Stepper.h
├── README.md
├── AutomatedRailwayCrossing.ino
text

## Components

- Arduino
- Stepper Motor
- Ultrasonic Sensors
- Red Light (LED)
- Buzzer
- Emergency Button
- Solar Panel
- Battery

## Setup

1. Connect the components as described in the respective header files.
2. Upload the `AutomatedRailwayCrossing.ino` sketch to your Arduino.
3. Open the Serial Monitor to observe the system status.

## Usage

- The system will automatically detect approaching trains and close the barrier.
- The warning lights and buzzer will activate when a train is detected.
- An emergency button can be used to override the system and open the barrier.

## Notes

- Adjust sensor thresholds and stepper motor settings as needed.
- Ensure proper power management for outdoor use.

Created on 2010-12-21