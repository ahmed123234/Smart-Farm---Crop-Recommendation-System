import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
MODEL_FILE = 'crop_recommendation_model_pipeline.pkl'

# Initialize Flask application
app = Flask(__name__)

# --- Load Model ---
# Load the pre-trained model 
try:
    with open(MODEL_FILE, 'rb') as file:
        pipeline = pickle.load(file)
    print(f"Successfully loaded model pipeline from {MODEL_FILE}")
except FileNotFoundError:
    # IMPORTANT: Ensure 'train.py' has been run to create this file!
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    print("Please run 'python train.py' first to train and save the model.")
    pipeline = None
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

# Placeholder for the exact feature order and structure expected by the pipeline
# This list MUST match the columns used in 'train.py' for non-encoded features.
EXPECTED_RAW_COLUMNS = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
]


# --- API Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    # 1. Get data from POST request
    data = request.get_json(force=True)
    
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

    # Ensure the input is structured as a DataFrame with expected columns
    try:
        # Extract features from the request data
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        row_data = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
   
        # # Create a DataFrame from the incoming data dictionary
        input_df = pd.DataFrame([row_data], columns=EXPECTED_RAW_COLUMNS)
    
    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {e}'}), 400

    except ValueError as e:
        return jsonify({"error": f"Input data structure error: {e}. Check missing columns."}), 400


    # Scale the input data using the loaded scaler
    # scaled_data = scaler.transform(input_data)

    # # 2. Make Prediction
    probability_of_recommended_crop = pipeline.predict_proba(input_df)[:, 1][0]
    
    # # Predict the final class
    prediction = pipeline.predict(input_df)[0]
    
    # 3. Return result as JSON
    response = {
        'confidence_score': round(probability_of_recommended_crop, 4),
        'recommended_crop': prediction
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Crop Recommendation Prediction Service is running. Use POST /predict to submit data."

# --- Run Server ---
if __name__ == '__main__':
    
    # Running on all interfaces (0.0.0.0) and port 5000 is standard for deployment
    app.run(host='0.0.0.0', port=5000)
