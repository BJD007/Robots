#include "Barrier.h"

Barrier::Barrier(int stepsPerRevolution, int in1, int in2, int in3, int in4)
    : stepper(stepsPerRevolution, in1, in2, in3, in4), barrierOpen(true) {
    stepper.setSpeed(10); // Adjust as needed
}

void Barrier::close() {
    if (barrierOpen) {
        Serial.println("Closing barrier");
        stepper.step(stepper.stepsPerRevolution() / 4);
        barrierOpen = false;
    }
}

void Barrier::open() {
    if (!barrierOpen) {
        Serial.println("Opening barrier");
        stepper.step(-stepper.stepsPerRevolution() / 4);
        barrierOpen = true;
    }
}

bool Barrier::isOpen() const {
    return barrierOpen;
}
