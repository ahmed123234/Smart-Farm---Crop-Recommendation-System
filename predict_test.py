import requests
import json

# URL of the Flask endpoint
URL = 'http://127.0.0.1:9040/predict' # Use 127.0.0.1 if running locally

# Sample data to send (MUST contain all 7 features)

data = {
    "N": 1,
    "P": 6,
    "K": 35,
    "temperature": 27.0227,
    "humidity": 95.7194,
    "ph": 6.2317,
    "rainfall": 147.1682
}

# Send the POST request
try:
    response = requests.post(URL, json=data)

    # Check the status code first!
    if response.status_code == 200:
        print("SUCCESS: Prediction received.")
        print(response.json())
    else:
        # If status is not 200, print the status and the raw text received
        print(f"ERROR: Request failed with status code {response.status_code}")
        print("--- RAW RESPONSE TEXT ---")
        print(response.text)
        print("-------------------------")

except requests.exceptions.ConnectionError:
    print(f"ERROR: Could not connect to the server at {URL}. Is 'python predict.py' running?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")