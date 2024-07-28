#ifndef BARRIER_H
#define BARRIER_H

#include <Arduino.h>
#include <Stepper.h>

class Barrier {
public:
    Barrier(int stepsPerRevolution, int in1, int in2, int in3, int in4);
    void close();
    void open();
    bool isOpen() const;

private:
    Stepper stepper;
    bool barrierOpen;
};

#endif
