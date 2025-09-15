import numpy as np
import os
import joblib
import re
from django.conf import settings  # if used in Django

MODEL_STORE = os.path.join(settings.BASE_DIR, "models_store")

# Load model and encoders
model = joblib.load(os.path.join(MODEL_STORE, "crop_yield_model.pkl"))
crop_encoder = joblib.load(os.path.join(MODEL_STORE, "Crop_encoder.pkl"))      # for 'Crop'
state_encoder = joblib.load(os.path.join(MODEL_STORE, "State_encoder.pkl"))    # for 'State'
season_encoder = joblib.load(os.path.join(MODEL_STORE, "Season_encoder.pkl"))  # for 'Season'

# Input sanitization
def sanitize_numeric_input(value, min_val=None, max_val=None, field_name=""):
    try:
        cleaned = re.sub(r"[^0-9.-]", "", str(value))
        num_value = float(cleaned)
        if min_val is not None and num_value < min_val:
            raise ValueError(f"{field_name} must be at least {min_val}")
        if max_val is not None and num_value > max_val:
            raise ValueError(f"{field_name} must be at most {max_val}")
        return num_value
    except ValueError as e:
        raise ValueError(f"Invalid {field_name}: {str(e)}")

def sanitize_input(text, max_length=255):
    if not isinstance(text, str):
        return ""
    return text.strip()[:max_length]

# Prediction function
def predict_yield(data: dict):
    try:
        # Required fields in input JSON
        required_fields = ["area", "item", "season", "crop_year", "average_rainfall", "pesticides", "annual_rainfall"]
        for field in required_fields:
            if field not in data:
                return {"success": False, "error": f"Missing field: {field}"}

        # Sanitize inputs
        crop = sanitize_input(data["item"])
        season = sanitize_input(data["season"])
        state = sanitize_input(data["area"])
        crop_year = sanitize_numeric_input(data["crop_year"], 2000, 2100, "Crop Year")
        avg_rainfall = sanitize_numeric_input(data["average_rainfall"], 0, 10000, "Average Rainfall")
        pesticides = sanitize_numeric_input(data["pesticides"], 0, 10000, "Pesticides")
        annual_rainfall = sanitize_numeric_input(data["annual_rainfall"], 0, 20000, "Annual Rainfall")

        # Encode categorical features
        crop_encoded = crop_encoder.transform([crop])[0]
        season_encoded = season_encoder.transform([season])[0]
        state_encoded = state_encoder.transform([state])[0]

        # Build feature array in order expected by model
        features = np.array([[crop_encoded, crop_year, season_encoded, state_encoded,
                              avg_rainfall, pesticides, annual_rainfall]])

        # Predict
        prediction = float(round(model.predict(features)[0], 2))
        return {"success": True, "prediction": prediction}

    except Exception as e:
        return {"success": False, "error": str(e)}
