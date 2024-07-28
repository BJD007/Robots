#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

class Sensors {
public:
    Sensors(int trigPin1, int echoPin1, int trigPin2, int echoPin2);
    int measureDistance(int trigPin, int echoPin);
    bool isTrainApproaching();

private:
    int trigPin1, echoPin1, trigPin2, echoPin2;
};

#endif
