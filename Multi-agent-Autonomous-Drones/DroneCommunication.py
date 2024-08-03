#MQTT Client: The DroneCommunication class sets up an MQTT client for each drone, connecting to the broker and handling message reception.
#Publish/Subscribe: Drones publish their data to specific topics and subscribe to control topics for receiving commands.
#JSON Serialization: Data is serialized using JSON for efficient transmissioni

mport paho.mqtt.client as mqtt
import json

class DroneCommunication:
    def __init__(self, broker_address, drone_id):
        self.client = mqtt.Client(drone_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker_address = broker_address

    def connect(self):
        self.client.connect(self.broker_address, 1883, 60)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("drone/control")
        client.subscribe("drone/data")

    def on_message(self, client, userdata, msg):
        print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")
        data = json.loads(msg.payload.decode())
        # Process incoming data

    def send_data(self, topic, data):
        self.client.publish(topic, json.dumps(data))

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

# Sample usage
drone_communication = DroneCommunication("broker.hivemq.com", "drone1")
drone_communication.connect()

# Send data to control station
drone_communication.send_data("drone/data", {"position": [5, 5], "battery": 85})

# Disconnect when done
drone_communication.disconnect()
