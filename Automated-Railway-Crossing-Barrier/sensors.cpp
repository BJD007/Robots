#include "Sensors.h"

Sensors::Sensors(int trigPin1, int echoPin1, int trigPin2, int echoPin2)
    : trigPin1(trigPin1), echoPin1(echoPin1), trigPin2(trigPin2), echoPin2(echoPin2) {
    pinMode(trigPin1, OUTPUT);
    pinMode(echoPin1, INPUT);
    pinMode(trigPin2, OUTPUT);
    pinMode(echoPin2, INPUT);
}

int Sensors::measureDistance(int trigPin, int echoPin) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    long duration = pulseIn(echoPin, HIGH);
    return duration * 0.034 / 2;
}

bool Sensors::isTrainApproaching() {
    int distance1 = measureDistance(trigPin1, echoPin1);
    int distance2 = measureDistance(trigPin2, echoPin2);
    return (distance1 < 100 || distance2 < 100);
}
