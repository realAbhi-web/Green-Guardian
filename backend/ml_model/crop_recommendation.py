import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + label encoder once
model_path = os.path.join(BASE_DIR, "../models_store/crop_recommendation.pkl")
encoder_path = os.path.join(BASE_DIR, "../models_store/crop_recommendation_label_encoder.pkl")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_num = model.predict(features)[0]
    crop = label_encoder.inverse_transform([prediction_num])[0]
    return crop
