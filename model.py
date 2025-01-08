import numpy as np

def predict_sales(input_data):
    # Simple placeholder prediction logic
    # Replace this with your actual ML model
    total = (
        input_data['traffic'] * 0.3 +
        input_data['marketing'] * 0.25 +
        input_data['advertising'] * 0.2 +
        input_data['social'] * 0.15
    )
    return total 