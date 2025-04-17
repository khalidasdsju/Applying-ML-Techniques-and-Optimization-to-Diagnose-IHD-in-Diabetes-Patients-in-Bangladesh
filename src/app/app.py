"""
Web application for IHD diagnosis project.
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__, template_folder='templates')

# Load model
MODEL_PATH = os.path.join('models', 'voting_ensemble.joblib')
model = None

# Load label encoders and scaler
LABEL_ENCODERS_PATH = os.path.join('models', 'label_encoders.joblib')
SCALER_PATH = os.path.join('models', 'scaler.joblib')
label_encoders = None
scaler = None


def load_model():
    """
    Load model, label encoders, and scaler.

    Returns:
    --------
    tuple
        Model, label encoders, and scaler.
    """
    global model, label_encoders, scaler

    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        else:
            print(f"Model file not found: {MODEL_PATH}")
            model = None

        # Load label encoders
        if os.path.exists(LABEL_ENCODERS_PATH):
            label_encoders = joblib.load(LABEL_ENCODERS_PATH)
            print(f"Loaded label encoders from {LABEL_ENCODERS_PATH}")
        else:
            print(f"Label encoders file not found: {LABEL_ENCODERS_PATH}")
            label_encoders = None

        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Loaded scaler from {SCALER_PATH}")
        else:
            print(f"Scaler file not found: {SCALER_PATH}")
            scaler = None

        return model, label_encoders, scaler

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def preprocess_input(input_data):
    """
    Preprocess input data.

    Parameters:
    -----------
    input_data : dict
        Input data.

    Returns:
    --------
    pd.DataFrame
        Preprocessed data.
    """
    # Convert input data to dataframe
    df = pd.DataFrame([input_data])

    # Encode categorical features
    if label_encoders is not None:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

    # Scale numerical features
    if scaler is not None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


def make_prediction(input_data):
    """
    Make prediction.

    Parameters:
    -----------
    input_data : dict
        Input data.

    Returns:
    --------
    dict
        Prediction result.
    """
    # Check if model is loaded
    if model is None:
        load_model()
        if model is None:
            return {"error": "Model not loaded"}

    try:
        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(preprocessed_data)[0]

        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(preprocessed_data)[0]
            probability = probabilities.max()
        else:
            probability = None

        # Create result
        result = {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def home():
    """
    Home page.

    Returns:
    --------
    str
        Rendered template.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.

    Returns:
    --------
    dict
        Prediction result.
    """
    # Get input data
    input_data = request.json

    # Make prediction
    result = make_prediction(input_data)

    return jsonify(result)


@app.route('/health')
def health():
    """
    Health check endpoint.

    Returns:
    --------
    dict
        Health status.
    """
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    # Load model
    load_model()

    # Run app
    app.run(debug=True, host='0.0.0.0', port=8080)
