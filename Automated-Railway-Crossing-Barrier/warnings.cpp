#include "Warnings.h"

Warnings::Warnings(int redLightPin, int buzzerPin)
    : redLightPin(redLightPin), buzzerPin(buzzerPin) {
    pinMode(redLightPin, OUTPUT);
    pinMode(buzzerPin, OUTPUT);
}

void Warnings::activate() {
    digitalWrite(redLightPin, HIGH);
    tone(buzzerPin, 1000); // 1kHz warning tone
}

void Warnings::deactivate() {
    digitalWrite(redLightPin, LOW);
    noTone(buzzerPin);
}
