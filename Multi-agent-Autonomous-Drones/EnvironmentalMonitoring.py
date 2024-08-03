#Sensor Data Analysis: Simple analysis based on threshold values for temperature, humidity, and air quality.
#Real-time Processing: The module processes data onboard for immediate decision-making.


class EnvironmentalMonitoring:
    def __init__(self, sensor_data):
        self.sensor_data = sensor_data

    def analyze_data(self):
        temperature = self.sensor_data['temperature']
        humidity = self.sensor_data['humidity']
        air_quality = self.sensor_data['air_quality']

        # Simple analysis example
        if temperature > 30 and humidity < 40:
            print("Warning: High temperature and low humidity!")
        if air_quality < 50:
            print("Air quality is good.")
        else:
            print("Air quality is poor.")

# Sample usage
sensor_data = {'temperature': 32, 'humidity': 35, 'air_quality': 45}
environmental_monitoring = EnvironmentalMonitoring(sensor_data)
environmental_monitoring.analyze_data()
