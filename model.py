import numpy as np
import joblib
from typing import Dict, Any

def convert_numpy_float(value: Any) -> float:
    """Convert numpy float types to Python float."""
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    return value

def predict_sales(input_data: Dict[str, Any]) -> float:
    """
    Predict sales based on the selected machine learning model.
    Returns prediction as a Python float.
    """
    try:
        model_key = input_data["ml_model"]
        if model_key not in models:
            raise ValueError(f"Model '{model_key}' is not available.")

        # Extract features and reshape for prediction
        features = np.array([[
            input_data['traffic'],
            input_data['marketing'],
            input_data['advertising'],
            input_data['social']
        ]], dtype=np.float64)
        
        # Use the selected model for prediction
        model = models[model_key]
        prediction = model.predict(features)
        
        # Convert numpy float32/float64 to Python float
        return convert_numpy_float(prediction[0])
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Load pre-trained models from the 'models/' directory
models = {
    "random_forest": joblib.load("models/random_forest.joblib"),
    "gradient_boost": joblib.load("models/gradient_boost.joblib"),
    "xgboost": joblib.load("models/xgboost.joblib"),
    "lightgbm": joblib.load("models/lightgbm.joblib"),
    "svr": joblib.load("models/svr.joblib"),
    "neural_network": joblib.load("models/neural_network.joblib"),
}
