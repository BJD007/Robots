import RPi.GPIO as GPIO
import Adafruit_ADS1x15
import cv2

def initialize_sensors():
    GPIO.setmode(GPIO.BCM)
    
    sensors = {
        'flame': {'pin': 17, 'type': 'digital'},
        'gas': {'pin': 18, 'type': 'digital'},
        'soil_moisture': {'pin': 27, 'type': 'analog'},
        'human': {'pin': 22, 'type': 'digital'},
        'adc': Adafruit_ADS1x15.ADS1115(),
        'camera': cv2.VideoCapture(0)
    }

    for sensor in sensors.values():
        if isinstance(sensor, dict) and sensor['type'] == 'digital':
            GPIO.setup(sensor['pin'], GPIO.IN)

    return sensors

def read_sensor(sensor):
    if sensor['type'] == 'digital':
        return GPIO.input(sensor['pin'])
    elif sensor['type'] == 'analog':
        return sensors['adc'].read_adc(sensor['pin'], gain=1)

def read_all_sensors(sensors, vehicle):
    return {
        'flame': read_sensor(sensors['flame']),
        'gas': read_sensor(sensors['gas']),
        'soil_moisture': read_sensor(sensors['soil_moisture']),
        'human': read_sensor(sensors['human']),
        'gps': {
            'lat': vehicle.location.global_frame.lat,
            'lon': vehicle.location.global_frame.lon,
            'alt': vehicle.location.global_frame.alt
        },
        'battery': vehicle.battery.level
    }
