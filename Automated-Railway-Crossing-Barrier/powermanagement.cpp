#include "PowerManagement.h"

PowerManagement::PowerManagement(int solarPanelPin, int batteryLevelPin)
    : solarPanelPin(solarPanelPin), batteryLevelPin(batteryLevelPin) {
    pinMode(solarPanelPin, INPUT);
    pinMode(batteryLevelPin, INPUT);
}

void PowerManagement::checkLevels() {
    int solarPanelValue = analogRead(solarPanelPin);
    int batteryLevel = analogRead(batteryLevelPin);
    
    if (solarPanelValue > 700) {
        Serial.println("Using solar power");
    } else if (batteryLevel < 300) {
        Serial.println("Low battery! Entering power saving mode");
        // Implement power saving measures here
    }
}
