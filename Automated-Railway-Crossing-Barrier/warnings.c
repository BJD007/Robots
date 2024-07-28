#ifndef WARNINGS_H
#define WARNINGS_H

#include <Arduino.h>

class Warnings {
public:
    Warnings(int redLightPin, int buzzerPin);
    void activate();
    void deactivate();

private:
    int redLightPin, buzzerPin;
};

#endif

