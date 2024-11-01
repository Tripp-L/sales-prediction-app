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
from datetime import datetime
import logging

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
        'advertising_budget': np.random.uniform(1000, 10000, n_samples),
        'social_media_spend': np.random.uniform(200, 2000, n_samples),
        'date': pd.date_range(start='2023-01-01', periods=n_samples)
    })
    
    # Create seasonal effects
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int)
    
    # Generate target variable (sales)
    seasonal_effect = 2000 * np.sin(2 * np.pi * data['date'].dt.dayofyear / 365)
    
    data['sales'] = (
        0.3 * data['website_traffic'] +
        0.5 * data['marketing_spend'] +
        0.4 * data['advertising_budget'] +
        0.2 * data['social_media_spend'] +
        seasonal_effect +
        np.random.normal(0, 1000, n_samples)
    )
    
    return data

def create_gauge_chart(prediction, max_value):
    """Create gauge chart for prediction visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Sales ($)", 'font': {'color': "#2ecc71", 'size': 24}},
        gauge = {
            'axis': {'range': [0, max_value], 'tickcolor': "#2ecc71"},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#2ecc71",
            'steps': [
                {'range': [0, max_value/3], 'color': '#1a1a1a'},
                {'range': [max_value/3, max_value*2/3], 'color': '#262626'},
                {'range': [max_value*2/3, max_value], 'color': '#333333'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#2ecc71", 'family': "Arial"}
    )
    
    return fig

def create_sales_trend_plot(data):
    """Create a sales trend visualization"""
    fig = go.Figure()
    
    # Add sales trend line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['sales'],
        name='Historical Sales',
        line=dict(color='#2ecc71', width=2),
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sales Trend Over Time',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#2ecc71')
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title='Sales ($)',
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(color='#e0e0e0')
        )
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
        model = trained_models[model_type]
        prediction = float(model.predict(input_scaled)[0])
        
        # Calculate confidence score (simplified)
        confidence_score = 0.9  # Placeholder
        
        # Create gauge chart
        max_value = prediction * 1.5
        fig = create_gauge_chart(prediction, max_value)
        
        return jsonify({
            'prediction': prediction,
            'accuracy': model.score(X_test_scaled, y_test),
            'confidence': confidence_score,
            'plot': fig.to_dict()
        })
        
    except Exception as e:
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
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
