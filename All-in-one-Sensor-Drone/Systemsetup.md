Based on the information provided, you are using a Pixhawk 4 flight controller with the PX4 autopilot software for your custom drone project. Here are some key points about using the Pixhawk autopilot system:

1. Hardware:
- The Pixhawk 4 is a popular and capable flight controller that supports PX4 firmware.
- It has an STM32F765 processor, multiple IMUs, and various ports for connecting sensors and peripherals.

2. Software:
- PX4 is an open-source autopilot software that runs on the Pixhawk hardware.
- It provides features like stabilization, autonomous flight modes, mission planning, etc. 

3. Sensors:
- The Pixhawk 4 has built-in IMUs (accelerometer, gyroscope), barometer and magnetometer.
- You can connect external GPS, optical flow, lidar and other sensors.

4. Setup:
- Use QGroundControl software for initial setup, calibration and configuration.
- You'll need to calibrate sensors, set up flight modes, tune PID parameters, etc.

5. Connectivity:
- Pixhawk supports various telemetry options like radio, WiFi, 4G for communication with ground station.

6. Customization:
- PX4 is highly customizable - you can modify parameters, add custom flight modes, etc.

7. Support:
- There's extensive documentation and an active community for help with Pixhawk/PX4 setup.

For your GPS issue, double check the wiring and make sure you're using a compatible GPS module. You may need to configure the correct serial port in the PX4 parameters for the GPS.
