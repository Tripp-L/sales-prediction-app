import numpy as np
import joblib

# Load pre-trained models from the 'models/' directory
models = {
    "random_forest": joblib.load("models/random_forest.joblib"),
    "gradient_boost": joblib.load("models/gradient_boost.joblib"),
    "xgboost": joblib.load("models/xgboost.joblib"),
    "lightgbm": joblib.load("models/lightgbm.joblib"),
    "svr": joblib.load("models/svr.joblib"),
    "neural_network": joblib.load("models/neural_network.joblib"),
}

def predict_sales(input_data):
    """
    Predict sales based on the selected machine learning model.
    """
    model_key = input_data["ml_model"]
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' is not available.")

    # Extract only the features the model was trained on
    features = np.array([[input_data['traffic'], input_data['marketing'], 
                          input_data['advertising'], input_data['social']]])
    
    # Use the selected model for prediction
    model = models[model_key]
    prediction = model.predict(features)
    return prediction[0]
