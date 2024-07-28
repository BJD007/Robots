#include <XBee.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <NewPing.h>
#include <avr/sleep.h>
#include <avr/power.h>

// Pin definitions
#define XBEE_RX 2
#define XBEE_TX 3
#define SOIL_MOISTURE_PIN A0
#define ULTRASONIC_TRIGGER_PIN 7
#define ULTRASONIC_ECHO_PIN 8
#define DC_MOTOR_LEFT_PIN1 9
#define DC_MOTOR_LEFT_PIN2 10
#define DC_MOTOR_RIGHT_PIN1 11
#define DC_MOTOR_RIGHT_PIN2 12
#define STEPPER_PIN1 4
#define STEPPER_PIN2 5
#define STEPPER_PIN3 6
#define STEPPER_PIN4 13
#define BATTERY_VOLTAGE_PIN A1
#define BME280_POWER_PIN 14
#define ULTRASONIC_POWER_PIN 15
#define XBEE_SLEEP_PIN 16

// Constants
#define MAX_DISTANCE 200 // Maximum distance for ultrasonic sensor (in cm)
#define BATTERY_THRESHOLD 11.5 // Low battery threshold (in volts)

// Objects
XBee xbee = XBee();
SoftwareSerial xbeeSerial(XBEE_RX, XBEE_TX);
Adafruit_BME280 bme;
NewPing sonar(ULTRASONIC_TRIGGER_PIN, ULTRASONIC_ECHO_PIN, MAX_DISTANCE);

// Variables
float temperature, humidity, pressure;
int soilMoisture;
float batteryVoltage;
int distance;
bool inPowerSavingMode = false;

// XBee address (replace with your coordinator's address)
XBeeAddress64 addr64 = XBeeAddress64(0x0013A200, 0x40C6A1B1);

void setup() {
  Serial.begin(9600);
  xbeeSerial.begin(9600);
  xbee.setSerial(xbeeSerial);
  
  // Initialize sensors
  if (!bme.begin(0x76)) {
    Serial.println("Could not find a valid BME280 sensor, check wiring!");
    while (1);
  }
  
  // Initialize motor pins
  pinMode(DC_MOTOR_LEFT_PIN1, OUTPUT);
  pinMode(DC_MOTOR_LEFT_PIN2, OUTPUT);
  pinMode(DC_MOTOR_RIGHT_PIN1, OUTPUT);
  pinMode(DC_MOTOR_RIGHT_PIN2, OUTPUT);
  pinMode(STEPPER_PIN1, OUTPUT);
  pinMode(STEPPER_PIN2, OUTPUT);
  pinMode(STEPPER_PIN3, OUTPUT);
  pinMode(STEPPER_PIN4, OUTPUT);
  
  // Initialize power management pins
  pinMode(BATTERY_VOLTAGE_PIN, INPUT);
  pinMode(BME280_POWER_PIN, OUTPUT);
  pinMode(ULTRASONIC_POWER_PIN, OUTPUT);
  pinMode(XBEE_SLEEP_PIN, OUTPUT);
  
  digitalWrite(BME280_POWER_PIN, HIGH);
  digitalWrite(ULTRASONIC_POWER_PIN, HIGH);
  digitalWrite(XBEE_SLEEP_PIN, LOW);
  
  Serial.println("Agricultural Robot Initialized");
}

void loop() {
  if (!inPowerSavingMode) {
    readSensors();
    autonomousNavigation();
    sendData();
    checkBattery();
  } else {
    // In power saving mode, only perform minimal tasks
    wakeAndCheck();
  }
  delay(1000);
}

void readSensors() {
  temperature = bme.readTemperature();
  humidity = bme.readHumidity();
  pressure = bme.readPressure() / 100.0F;
  soilMoisture = analogRead(SOIL_MOISTURE_PIN);
  distance = sonar.ping_cm();
}

void autonomousNavigation() {
  if (distance < 30) {
    // Obstacle detected, turn
    turnRight();
  } else {
    // No obstacle, move forward
    moveForward();
  }
  
  // Check soil moisture and perform action if needed
  if (soilMoisture < 300) {
    // Soil is dry, perform watering action
    waterPlants();
  }
}

void moveForward() {
  digitalWrite(DC_MOTOR_LEFT_PIN1, HIGH);
  digitalWrite(DC_MOTOR_LEFT_PIN2, LOW);
  digitalWrite(DC_MOTOR_RIGHT_PIN1, HIGH);
  digitalWrite(DC_MOTOR_RIGHT_PIN2, LOW);
}

void turnRight() {
  digitalWrite(DC_MOTOR_LEFT_PIN1, HIGH);
  digitalWrite(DC_MOTOR_LEFT_PIN2, LOW);
  digitalWrite(DC_MOTOR_RIGHT_PIN1, LOW);
  digitalWrite(DC_MOTOR_RIGHT_PIN2, HIGH);
}

void waterPlants() {
  for (int i = 0; i < 200; i++) {
    digitalWrite(STEPPER_PIN1, HIGH);
    delay(10);
    digitalWrite(STEPPER_PIN1, LOW);
    digitalWrite(STEPPER_PIN2, HIGH);
    delay(10);
    digitalWrite(STEPPER_PIN2, LOW);
    digitalWrite(STEPPER_PIN3, HIGH);
    delay(10);
    digitalWrite(STEPPER_PIN3, LOW);
    digitalWrite(STEPPER_PIN4, HIGH);
    delay(10);
    digitalWrite(STEPPER_PIN4, LOW);
  }
}

void sendData() {
  String data = "T:" + String(temperature) + ",H:" + String(humidity) + 
                ",P:" + String(pressure) + ",SM:" + String(soilMoisture) + 
                ",D:" + String(distance) + ",B:" + String(batteryVoltage);
  
  uint8_t payload[data.length() + 1];
  data.getBytes(payload, data.length() + 1);
  
  ZBTxRequest zbTx = ZBTxRequest(addr64, payload, sizeof(payload));
  xbee.send(zbTx);
}

void checkBattery() {
  int sensorValue = analogRead(BATTERY_VOLTAGE_PIN);
  batteryVoltage = sensorValue * (5.0 / 1023.0) * 3; // Adjust multiplier based on voltage divider
  
  if (batteryVoltage < BATTERY_THRESHOLD) {
    // Low battery, send alert and enter power-saving mode
    String alert = "LOW_BATTERY";
    uint8_t payload[alert.length() + 1];
    alert.getBytes(payload, alert.length() + 1);
    
    ZBTxRequest zbTx = ZBTxRequest(addr64, payload, sizeof(payload));
    xbee.send(zbTx);
    
    enterPowerSavingMode();
  }
}

void enterPowerSavingMode() {
  if (!inPowerSavingMode) {
    inPowerSavingMode = true;
    
    // Turn off non-essential components
    digitalWrite(BME280_POWER_PIN, LOW);
    digitalWrite(ULTRASONIC_POWER_PIN, LOW);
    
    // Put XBee into sleep mode
    digitalWrite(XBEE_SLEEP_PIN, HIGH);
    
    // Disable ADC
    ADCSRA = 0;
    
    // Disable unused peripherals
    power_adc_disable();
    power_spi_disable();
    power_twi_disable();
    power_timer1_disable();
    power_timer2_disable();
    
    // Set all unused pins as inputs with pull-up resistors
    for (int i = 0; i <= A5; i++) {
      if (i != SOIL_MOISTURE_PIN && i != BATTERY_VOLTAGE_PIN) {
        pinMode(i, INPUT_PULLUP);
      }
    }
    
    // Reduce clock speed
    CLKPR = _BV(CLKPCE);
    CLKPR = _BV(CLKPS1) | _BV(CLKPS0); // Set clock divisor to 8
    
    // Set up watchdog timer for periodic wake-ups
    setupWatchdogTimer();
    
    // Enter sleep mode
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    sleep_enable();
    
    // Send notification that power saving mode is active
    sendPowerSavingNotification();
    
    Serial.println("Entering power saving mode");
    Serial.flush(); // Ensure last message is sent before sleep
    
    // Actually enter sleep mode
    sleep_mode();
  }
}

void setupWatchdogTimer() {
  // Clear the reset flag
  MCUSR &= ~(1<<WDRF);
  
  // Set up WDT interrupt
  WDTCSR = (1<<WDCE) | (1<<WDE);
  WDTCSR = (1<<WDIE) | (0<<WDE) | (1<<WDP3) | (0<<WDP2) | (0<<WDP1) | (1<<WDP0);
  // This sets the watchdog timer to interrupt every 8 seconds
}

ISR(WDT_vect) {
  // Watchdog timer interrupt handler
  // Wake up, perform minimal tasks, then go back to sleep
  wakeAndCheck();
}

void wakeAndCheck() {
  // Temporarily wake up to check battery and send status
  sleep_disable();
  
  // Re-enable necessary peripherals
  power_adc_enable();
  
  // Check battery
  checkBattery();
  
  // Send minimal status update
  sendMinimalStatusUpdate();
  
  // Go back to sleep if still in power saving mode
  if (inPowerSavingMode) {
    sleep_enable();
    sleep_mode();
  }
}

void sendPowerSavingNotification() {
  String message = "STATUS:POWER_SAVING_MODE";
  
  uint8_t payload[message.length() + 1];
  message.getBytes(payload, message.length() + 1);
  
  ZBTxRequest zbTx = ZBTxRequest(addr64, payload, sizeof(payload));
  
  // Wake up XBee momentarily to send the message
  digitalWrite(XBEE_SLEEP_PIN, LOW);
  delay(15); // Give XBee time to wake up
  
  xbee.send(zbTx);
  
  // Wait for the message to be sent
  delay(100);
  
  // Put XBee back to sleep
  digitalWrite(XBEE_SLEEP_PIN, HIGH);
  
  Serial.println("Sent power saving mode notification");
}

void sendMinimalStatusUpdate() {
  // Read battery voltage
  int sensorValue = analogRead(BATTERY_VOLTAGE_PIN);
  float batteryVoltage = sensorValue * (5.0 / 1023.0) * 3; // Adjust multiplier based on voltage divider
  
  String message = "STATUS:MINIMAL,BATT:" + String(batteryVoltage, 2);
  
  uint8_t payload[message.length() + 1];
  message.getBytes(payload, message.length() + 1);
  
  ZBTxRequest zbTx = ZBTxRequest(addr64, payload, sizeof(payload));
  
  // Wake up XBee momentarily to send the message
  digitalWrite(XBEE_SLEEP_PIN, LOW);
  delay(15); // Give XBee time to wake up
  
  xbee.send(zbTx);
  
  // Wait for the message to be sent
  delay(100);
  
  // Put XBee back to sleep
  digitalWrite(XBEE_SLEEP_PIN, HIGH);
  
  Serial.println("Sent minimal status update");
}

void exitPowerSavingMode() {
  if (inPowerSavingMode) {
    inPowerSavingMode = false;
    
    // Disable sleep mode
    sleep_disable();
    
    // Reset clock speed
    CLKPR = _BV(CLKPCE);
    CLKPR = 0; // Set clock divisor back to 1
    
    // Re-enable peripherals
    power_all_enable();
    
    // Re-enable ADC
    ADCSRA = 0x87;
    
    // Turn on components
    digitalWrite(BME280_POWER_PIN, HIGH);
    digitalWrite(ULTRASONIC_POWER_PIN, HIGH);
    
    // Wake up XBee
    digitalWrite(XBEE_SLEEP_PIN, LOW);
    
    // Reinitialize sensors and other components as needed
    initializeSensors();
    
    // Send notification that normal operation has resumed
    sendNormalOperationNotification();
    
    Serial.println("Exited power saving mode");
  }
}

void initializeSensors() {
  if (!bme.begin(0x76)) {
    Serial.println("Could not find a valid BME280 sensor, check wiring!");
  }
  // Add other sensor initializations as needed
}

void sendNormalOperationNotification() {
  String message = "STATUS:NORMAL_OPERATION";
  
  uint8_t payload[message.length() + 1];
  message.getBytes(payload, message.length() + 1);
  
  ZBTxRequest zbTx = ZBTxRequest(addr64, payload, sizeof(payload));
  
  digitalWrite(XBEE_SLEEP_PIN, LOW);
  delay(15); // Give XBee time to wake up if it was asleep
  
  xbee.send(zbTx);
  
  Serial.println("Sent normal operation notification");
}% Main File
