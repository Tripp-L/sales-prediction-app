from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from model import predict_sales  # Assuming you have this function

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_visualization(input_data, prediction):
    # Placeholder for visualization logic
    # Return empty string for now, implement actual visualization later
    return ""

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract input parameters
        input_data = {
            'traffic': float(data['traffic']),
            'marketing': float(data['marketing']),
            'advertising': float(data['advertising']),
            'social': float(data['social']),
            'other_expenses': float(data.get('other_expenses', 0)),
            'date': data['date'],
            'ml_model': data['ml_model']
        }
        
        # Make prediction
        prediction = predict_sales(input_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'accuracy_score': 0.85,  # Replace with actual model accuracy
            'confidence_score': 0.78,  # Replace with actual confidence score
            'plot': generate_visualization(input_data, prediction)  # Create this function
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
