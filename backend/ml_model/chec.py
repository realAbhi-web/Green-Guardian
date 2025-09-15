import joblib

model = joblib.load("models_store/crop_yield_model.pkl")

# Number of features the model expects
print(model.n_features_in_)

# Feature names, if saved during training
if hasattr(model, 'feature_names_in_'):
    print(model.feature_names_in_)

# You can also check the model parameters
print(model.get_params())
