# import joblib
# import pandas as pd
import os, joblib, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models_store', 'fertilizer_model.pkl'))
soil_encoder = joblib.load(os.path.join(BASE_DIR, 'models_store', 'soil_encoder.pkl'))
crop_encoder = joblib.load(os.path.join(BASE_DIR, 'models_store', 'fertilizer_recommendation_crop_encoder.pkl'))
fertilizer_encoder = joblib.load(os.path.join(BASE_DIR, 'models_store', 'fertilizer_encoder.pkl'))

# # Load model and encoders
# model = joblib.load('models_store/fertilizer_model.pkl')
# soil_encoder = joblib.load('models_store/soil_encoder.pkl')
# crop_encoder = joblib.load('models_store/fertilizer_recommendation_crop_encoder.pkl')
# fertilizer_encoder = joblib.load('models_store/fertilizer_encoder.pkl')

def decode_fertilizer(encoded_label):
    return fertilizer_encoder.inverse_transform([encoded_label])[0]

def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Encode categorical inputs
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    # Prepare input DataFrame
    input_data = pd.DataFrame([[
        temperature, humidity, moisture,
        soil_encoded, crop_encoded,
        nitrogen, potassium, phosphorous
    ]], columns=[
        'Temparature', 'Humidity ', 'Moisture',
        'Soil Type', 'Crop Type',
        'Nitrogen', 'Potassium', 'Phosphorous'
    ])

    # Predict
    pred = model.predict(input_data)[0]
    return decode_fertilizer(pred)
