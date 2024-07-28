import requests
import json
import config

def send_data_over_4g(data):
    try:
        # Configure the 4G connection (placeholder)
        # ...

        # Prepare the data for transmission
        json_data = json.dumps(data)

        # Send the data to the server
        response = requests.post(config.SERVER_URL, data=json_data, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error sending data over 4G: {str(e)}")

    finally:
        # Disconnect from the 4G network (placeholder)
        # ...
        pass
