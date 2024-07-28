#ifndef POWER_MANAGEMENT_H
#define POWER_MANAGEMENT_H

#include <Arduino.h>

class PowerManagement {
public:
    PowerManagement(int solarPanelPin, int batteryLevelPin);
    void checkLevels();

private:
    int solarPanelPin, batteryLevelPin;
};

#endif
