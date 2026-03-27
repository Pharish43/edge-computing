"""
Simple interface to predict crop from land parameters
This script loads the trained model and makes predictions
"""

import pickle
import numpy as np
import pandas as pd


def load_model():
    """Load the trained model and preprocessors"""
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('crop_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('crop_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Predict the best crop for given land parameters

    Args:
        nitrogen (float): Nitrogen content in soil (0-140)
        phosphorus (float): Phosphorus content in soil (5-145)
        potassium (float): Potassium content in soil
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage (0-100)
        ph (float): pH level of soil (3.5-9.9)
        rainfall (float): Rainfall in mm

    Returns:
        tuple: (crop_name, confidence_percentage)
    """
    model, scaler, encoder = load_model()

    # Create all engineered features
    n_p_ratio = nitrogen / (phosphorus + 1e-6)
    k_n_ratio = potassium / (nitrogen + 1e-6)
    p_k_ratio = phosphorus / (potassium + 1e-6)
    npk_sum = nitrogen + phosphorus + potassium
    npk_avg = npk_sum / 3
    npk_ratio = npk_sum / (rainfall + 1e-6)
    temp_humidity = temperature * humidity
    rainfall_temp = rainfall * temperature
    ph_neutral_dist = np.abs(ph - 6.5)
    hot_humid = 1 if (temperature > 25 and humidity > 75) else 0
    moderate_rainfall = 1 if (100 < rainfall < 250) else 0

    # Create DataFrame with correct feature names (as fitted in training)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
                     'N_P_ratio', 'K_N_ratio', 'P_K_ratio', 'NPK_sum', 'NPK_avg',
                     'NPK_ratio', 'temp_humidity', 'rainfall_temp', 'pH_neutral_dist',
                     'hot_humid', 'moderate_rainfall']

    feature_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall,
                      n_p_ratio, k_n_ratio, p_k_ratio, npk_sum, npk_avg,
                      npk_ratio, temp_humidity, rainfall_temp, ph_neutral_dist,
                      hot_humid, moderate_rainfall]

    # Create DataFrame with feature names
    input_data = pd.DataFrame([feature_values], columns=feature_names)

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Get prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Decode and get confidence
    crop_name = encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities)) * 100

    return crop_name, confidence


def interactive_prediction():
    """Interactive mode for crop prediction"""
    print("="*60)
    print("CROP RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nEnter your land parameters to predict suitable crops:\n")

    try:
        nitrogen = float(input("Nitrogen (N) [0-140]: "))
        phosphorus = float(input("Phosphorus (P) [5-145]: "))
        potassium = float(input("Potassium (K) [0-120]: "))
        temperature = float(input("Temperature (Celsius) [10-50]: "))
        humidity = float(input("Humidity (%) [0-100]: "))
        ph = float(input("pH level [3.5-9.9]: "))
        rainfall = float(input("Rainfall (mm) [20-300]: "))

        crop, confidence = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Recommended Crop: {crop.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*60)

    except ValueError:
        print("Error: Please enter valid numbers!")


if __name__ == "__main__":
    interactive_prediction()
