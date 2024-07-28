#include <Arduino.h>
#include "Barrier.h"
#include "Sensors.h"
#include "Warnings.h"
#include "PowerManagement.h"

#define STEPS 2048
#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11

Barrier barrier(STEPS, IN1, IN2, IN3, IN4);
Sensors sensors(2, 3, 4, 5);
Warnings warnings(6, 7);
PowerManagement powerManagement(A0, A1);

const int emergencyButtonPin = 12;
volatile bool emergencyOverride = false;

void emergencyStop() {
    emergencyOverride = !emergencyOverride;
    if (emergencyOverride) {
        Serial.println("Emergency override activated");
        warnings.deactivate();
        barrier.open();
    } else {
        Serial.println("Emergency override deactivated");
    }
}

void setup() {
    Serial.begin(9600);
    pinMode(emergencyButtonPin, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(emergencyButtonPin), emergencyStop, FALLING);
}

void loop() {
    powerManagement.checkLevels();
    
    if (!emergencyOverride) {
        if (sensors.isTrainApproaching() && barrier.isOpen()) {
            warnings.activate();
            barrier.close();
        } else if (!sensors.isTrainApproaching() && !barrier.isOpen()) {
            warnings.deactivate();
            barrier.open();
        }
    }
    
    delay(100);
}
