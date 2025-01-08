from flask import Flask, render_template, request, jsonify, send_file
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
import io
import plotly.express as px
from scipy import stats
import calendar
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
import json
from datetime import datetime, timedelta
import xlsxwriter
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
trained_models = {}
feature_scaler = None

# Update the model descriptions in your Flask app
MODEL_NOTES = {
    'random_forest': '* Ensemble learning method using multiple decision trees. Best for balanced performance and handling non-linear patterns.',
    'gradient_boost': '* Sequential ensemble method that builds trees to correct previous errors. High accuracy with proper tuning.',
    'xgboost': '* Advanced implementation of gradient boosting with better regularization and parallel processing.',
    'lightgbm': '* Gradient boosting framework using leaf-wise tree growth. Fast training and memory efficient.',
    'svr': '* Kernel-based method that maps data to higher dimensions. Effective for non-linear relationships.',
    'neural_network': '* Deep learning model with multiple layers. Automatically learns complex patterns from data.'
}

# Add this route to handle model selection changes
@app.route('/get_model_notes/<model_type>')
def get_model_notes(model_type):
    return jsonify({'notes': MODEL_NOTES.get(model_type, '')})

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

def create_sales_trend_plot(data, start_date=None, end_date=None, selected_metrics=None):
    """Create an enhanced sales trend visualization with interactive features"""
    if start_date:
        data = data[data['date'] >= start_date]
    if end_date:
        data = data[data['date'] <= end_date]
    
    fig = go.Figure()
    
    # Add sales trend line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['sales'],
        name='Sales',
        line=dict(color='#2ecc71', width=2),
        mode='lines'
    ))
    
    # Add additional metrics if selected
    if selected_metrics:
        colors = {'website_traffic': '#3498db', 
                 'marketing_spend': '#e74c3c', 
                 'advertising_budget': '#f1c40f',
                 'social_media_spend': '#9b59b6'}
        
        for metric in selected_metrics:
            if metric in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data[metric],
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors.get(metric, '#95a5a6'), width=2),
                    mode='lines',
                    visible='legendonly'  # Hidden by default
                ))
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': 'Interactive Sales Analytics Dashboard',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#2ecc71')
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Value',
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#2ecc71",
            borderwidth=1,
            font=dict(color='#e0e0e0')
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True] + [False] * (len(fig.data) - 1)}],
                        label="Sales Only",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                font=dict(color='#e0e0e0'),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#2ecc71"
            )
        ]
    )
    
    return fig

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug incoming request
        logger.debug(f"Request Headers: {dict(request.headers)}")
        logger.debug(f"Request Data: {request.get_data(as_text=True)}")

        # Check content type
        if not request.is_json:
            logger.error("Invalid Content-Type. Expected application/json")
            return jsonify({
                'error': 'Content-Type must be application/json',
                'received': request.headers.get('Content-Type')
            }), 415

        # Get JSON data
        data = request.get_json()
        logger.debug(f"Parsed JSON data: {data}")

        # Validate required fields
        required_fields = ['ml_model', 'forecast_period']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract values
        ml_model = data['ml_model']
        forecast_period = int(data['forecast_period'])

        # Log successful prediction request
        logger.info(f"Making prediction with model: {ml_model}, period: {forecast_period}")

        # Your prediction logic here
        predictions = [1, 2, 3]  # Replace with actual predictions

        return jsonify({
            'success': True,
            'predictions': predictions,
            'model': ml_model,
            'period': forecast_period
        })

    except Exception as e:
        logger.exception("Error processing prediction request")
        return jsonify({'error': str(e)}), 500

# Example prediction function (implement these according to your models)
def random_forest_predict(data, period):
    # Your prediction logic here
    pass

def gradient_boost_predict(data, period):
    # Your prediction logic here
    pass

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

@app.route('/export_data', methods=['POST'])
def export_data():
    try:
        format_type = request.form.get('format', 'csv')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        include_analysis = request.form.get('include_analysis') == 'true'
        
        # Filter data based on date range
        filtered_data = data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data['date'] >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data['date'] <= end_date]
        
        if format_type == 'csv':
            output = io.StringIO()
            filtered_data.to_csv(output, index=False)
            
            if include_analysis:
                analysis = perform_statistical_analysis(filtered_data)
                output.write('\n\nStatistical Analysis\n')
                output.write(json.dumps(analysis, indent=2))
            
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'sales_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            
        elif format_type == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_data.to_excel(writer, sheet_name='Data', index=False)
                
                if include_analysis:
                    analysis = perform_statistical_analysis(filtered_data)
                    pd.DataFrame([analysis['basic_stats']]).to_excel(
                        writer, sheet_name='Analysis', index=False)
                    
                    # Add charts
                    workbook = writer.book
                    chart_sheet = workbook.add_worksheet('Charts')
                    
                    # Sales trend chart
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name': 'Sales',
                        'categories': '=Data!$A$2:$A$' + str(len(filtered_data) + 1),
                        'values': '=Data!$B$2:$B$' + str(len(filtered_data) + 1),
                    })
                    chart_sheet.insert_chart('A1', chart)
            
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'sales_analysis_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
            
        elif format_type == 'json':
            output = {
                'data': filtered_data.to_dict(orient='records'),
                'analysis': perform_statistical_analysis(filtered_data) if include_analysis else None
            }
            return jsonify(output)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/filter_data', methods=['POST'])
def filter_data():
    try:
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        metrics = request.form.getlist('metrics')
        
        fig = create_sales_trend_plot(
            data, 
            start_date=start_date, 
            end_date=end_date,
            selected_metrics=metrics
        )
        
        return jsonify({
            'plot': fig.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_advanced_dashboard(data, start_date=None, end_date=None):
    """Create an interactive dashboard with multiple visualizations"""
    if start_date:
        data = data[data['date'] >= start_date]
    if end_date:
        data = data[data['date'] <= end_date]
    
    # Calculate key metrics
    daily_avg = data['sales'].mean()
    weekly_avg = data.resample('W', on='date')['sales'].mean().mean()
    growth_rate = ((data['sales'].iloc[-1] - data['sales'].iloc[0]) / data['sales'].iloc[0]) * 100
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Sales Trend', 'Sales Distribution',
            'Weekly Pattern', 'Monthly Pattern',
            'Correlation Matrix', 'Forecast'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "violin"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Sales Trend with Moving Averages
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['sales'], name='Daily Sales',
                  line=dict(color='#2ecc71', width=1)), row=1, col=1)
    
    # Add moving averages
    for window in [7, 30]:
        ma = data['sales'].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(x=data['date'], y=ma, name=f'{window}-day MA',
                      line=dict(width=2)), row=1, col=1)
    
    # 2. Sales Distribution
    fig.add_trace(
        go.Violin(y=data['sales'], name='Sales Distribution',
                 box_visible=True, line_color='#2ecc71',
                 meanline_visible=True), row=1, col=2)
    
    # 3. Weekly Pattern
    weekly_sales = data.groupby('day_of_week')['sales'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig.add_trace(
        go.Bar(x=days, y=weekly_sales, name='Avg by Day',
               marker_color='#2ecc71'), row=2, col=1)
    
    # 4. Monthly Pattern
    monthly_sales = data.groupby('month')['sales'].mean()
    months = [calendar.month_name[i] for i in range(1, 13)]
    fig.add_trace(
        go.Bar(x=months, y=monthly_sales, name='Avg by Month',
               marker_color='#2ecc71'), row=2, col=2)
    
    # 5. Correlation Matrix
    corr_matrix = data[['sales', 'website_traffic', 'marketing_spend', 
                       'advertising_budget', 'social_media_spend']].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns,
                   colorscale='Viridis'), row=3, col=1)
    
    # 6. Simple Forecast (last 30 days + 7 days forecast)
    forecast_days = 7
    last_30_days = data.tail(30).copy()
    last_30_days['days_from_start'] = range(30)
    model = LinearRegression()
    model.fit(last_30_days[['days_from_start']], last_30_days['sales'])
    
    future_dates = pd.date_range(
        start=data['date'].max() + timedelta(days=1),
        periods=forecast_days
    )
    future_days = range(30, 30 + forecast_days)
    forecast = model.predict([[x] for x in future_days])
    
    fig.add_trace(
        go.Scatter(x=last_30_days['date'], y=last_30_days['sales'],
                  name='Recent Sales', line=dict(color='#2ecc71')), row=3, col=2)
    fig.add_trace(
        go.Scatter(x=future_dates, y=forecast,
                  name='Forecast', line=dict(dash='dash')), row=3, col=2)
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': 'Advanced Sales Analytics Dashboard',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#2ecc71')
        }
    )
    
    # Add annotations for key metrics
    metrics_text = (
        f'Daily Average: ${daily_avg:,.2f}<br>'
        f'Weekly Average: ${weekly_avg:,.2f}<br>'
        f'Growth Rate: {growth_rate:.1f}%'
    )
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=1, y=1,
        showarrow=False,
        font=dict(size=12, color='#2ecc71'),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='#2ecc71',
        borderwidth=1,
        borderpad=10
    )
    
    return fig

def perform_statistical_analysis(data):
    """Perform advanced statistical analysis on sales data"""
    analysis = {
        'basic_stats': {
            'mean': data['sales'].mean(),
            'median': data['sales'].median(),
            'std': data['sales'].std(),
            'skew': data['sales'].skew(),
            'kurtosis': data['sales'].kurtosis()
        },
        'trend_analysis': {
            'growth_rate': ((data['sales'].iloc[-1] - data['sales'].iloc[0]) 
                           / data['sales'].iloc[0] * 100),
            'volatility': data['sales'].std() / data['sales'].mean() * 100
        }
    }
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data['sales'], period=7)
    analysis['seasonality'] = {
        'trend': decomposition.trend.dropna().tolist(),
        'seasonal': decomposition.seasonal.dropna().tolist(),
        'residual': decomposition.resid.dropna().tolist()
    }
    
    # Perform stationarity test
    adf_test = adfuller(data['sales'].dropna())
    analysis['stationarity'] = {
        'adf_statistic': adf_test[0],
        'p_value': adf_test[1],
        'is_stationary': adf_test[1] < 0.05
    }
    
    return analysis

@app.route('/update_visualization', methods=['POST'])
def update_visualization():
    try:
        viz_type = request.form.get('type', 'sales_trend')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        metrics = request.form.getlist('metrics')
        
        filtered_data = data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data['date'] >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data['date'] <= end_date]
            
        if viz_type == 'advanced_dashboard':
            fig = create_advanced_dashboard(filtered_data)
        else:
            fig = create_sales_trend_plot(filtered_data, selected_metrics=metrics)
            
        return jsonify({
            'plot': fig.to_dict(),
            'analysis': perform_statistical_analysis(filtered_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Add WebSocket support for real-time updates
from flask_socketio import SocketIO, emit
socketio = SocketIO(app)

def background_update():
    """Simulate real-time data updates"""
    while True:
        # Update with new data every minute
        new_data = generate_new_data_point()
        socketio.emit('data_update', {
            'timestamp': datetime.now().isoformat(),
            'data': new_data
        })
        time.sleep(60)

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to real-time updates'})

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
