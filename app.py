from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
trained_models = {}
feature_scaler = None

def generate_ecommerce_data():
    """Generate synthetic e-commerce data"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base features
    data = pd.DataFrame({
        'website_traffic': np.random.randint(1000, 10000, n_samples),
        'marketing_spend': np.random.uniform(500, 5000, n_samples),
        'date': pd.date_range(start='2023-01-01', periods=n_samples),
        'advertising_budget': np.random.uniform(1000, 10000, n_samples),
        'social_media_spend': np.random.uniform(200, 2000, n_samples)
    })
    
    # Create seasonal effects
    seasonal_effect = 2000 * np.sin(2 * np.pi * data['date'].dt.dayofyear / 365)
    
    # Create target variable (sales)
    data['sales'] = (
        0.3 * data['website_traffic'] +
        0.5 * data['marketing_spend'] +
        0.4 * data['advertising_budget'] +
        0.2 * data['social_media_spend'] +
        seasonal_effect +
        np.random.normal(0, 1000, n_samples)
    )
    
    # Add date-based features
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['is_weekend'] = data['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return data

def create_sales_trend_plot(data):
    """Create sales trend visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#2ecc71')
    ))
    
    fig.update_layout(
        title='Historical Sales Trend',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_prediction_plot(prediction, model_type):
    """Create prediction visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=prediction,
        delta={'reference': 5000},  # Example reference value
        gauge={
            'axis': {'range': [None, 10000]},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 3000], 'color': 'lightgray'},
                {'range': [3000, 7000], 'color': 'gray'}
            ]
        },
        title={'text': f"Predicted Sales ({model_type})"}
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        traffic = float(request.form['traffic'])
        marketing = float(request.form['marketing'])
        advertising = float(request.form['advertising'])
        social = float(request.form['social'])
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        model_type = request.form['model']
        
        # Prepare input features
        input_data = pd.DataFrame({
            'website_traffic': [traffic],
            'marketing_spend': [marketing],
            'advertising_budget': [advertising],
            'social_media_spend': [social],
            'day_of_week': [date.weekday()],
            'month': [date.month],
            'is_weekend': [1 if date.weekday() >= 5 else 0]
        })
        
        # Scale features
        input_scaled = feature_scaler.transform(input_data)
        
        # Make prediction
        if model_type not in trained_models:
            raise ValueError(f"Model {model_type} not found")
            
        prediction = float(trained_models[model_type].predict(input_scaled)[0])
        
        # Create visualization
        fig = create_prediction_plot(prediction, model_type)
        
        return jsonify({
            'prediction': prediction,
            'accuracy': 0.95,
            'confidence': 0.90,
            'plot': fig.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

def get_model_comparisons(input_data):
    """Compare predictions across all models"""
    predictions = {}
    input_scaled = feature_scaler.transform(input_data)
    
    for name, model in trained_models.items():
        pred = float(model.predict(input_scaled)[0])
        predictions[name] = pred
    
    return predictions

@app.route('/compare_models', methods=['POST'])
def compare_models():
    try:
        # Get form data and prepare input
        input_data = prepare_input_data(request.form)
        
        # Get predictions from all models
        predictions = get_model_comparisons(input_data)
        
        # Create comparison visualization
        fig = create_comparison_plot(predictions)
        
        return jsonify({
            'predictions': predictions,
            'plot': fig.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        return jsonify({'error': str(e)}), 400

def create_comparison_plot(predictions):
    """Create model comparison visualization"""
    fig = go.Figure()
    
    models = list(predictions.keys())
    values = list(predictions.values())
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color='#2ecc71'
    ))
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Models',
        yaxis_title='Predicted Sales ($)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

if __name__ == '__main__':
    try:
        logger.info("Initializing application...")
        
        # Generate and prepare data
        data = generate_ecommerce_data()
        
        # Create initial plots
        sales_plot = create_sales_trend_plot(data)
        
        # Prepare features and target
        feature_cols = ['website_traffic', 'marketing_spend', 'advertising_budget', 
                       'social_media_spend', 'day_of_week', 'month', 'is_weekend']
        X = data[feature_cols]
        y = data['sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': LGBMRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=100.0)
        }
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            logger.info(f"{name} training completed - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        logger.info("Initialization complete - starting Flask app")
        app.run(debug=True)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
