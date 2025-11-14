import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
MODEL_FILE = 'readmission_model_pipeline.pkl'

# Initialize Flask application
app = Flask(__name__)

# --- Load Model ---
# Load the pre-trained pipeline (Preprocessor + XGBoost Model)
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
# A small, incomplete sample is used here, but in a real app, this list is externalized.
EXPECTED_RAW_COLUMNS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'age', 'gender', 'race', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glipizide', 
    'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed'
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
        # Create a DataFrame from the incoming data dictionary
        input_df = pd.DataFrame([data], columns=EXPECTED_RAW_COLUMNS)
    except ValueError as e:
        return jsonify({"error": f"Input data structure error: {e}. Check missing columns."}), 400

    # 2. Make Prediction
    # The pipeline handles both preprocessing (scaling/encoding) and prediction
    # predict_proba returns probabilities for both classes [P(No Readmit), P(Readmit)]
    probability_of_readmission = pipeline.predict_proba(input_df)[:, 1][0]
    
    # Predict the final class (0 or 1)
    prediction = pipeline.predict(input_df)[0]
    
    # 3. Return result as JSON
    response = {
        'readmission_probability': round(probability_of_readmission, 4),
        'predicted_class': int(prediction),
        'message': 'High Risk of Readmission' if prediction == 1 else 'Low Risk of Readmission'
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Readmission Risk Prediction Service is running. Use POST /predict to submit data."

# --- Run Server ---
if __name__ == '__main__':
    # Running on all interfaces (0.0.0.0) and port 5000 is standard for deployment
    app.run(host='0.0.0.0', port=5000)
