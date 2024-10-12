from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load your data
data = pd.read_csv('sales_data.csv')
data['date'] = pd.to_datetime(data['date'])

# After loading your data
print(f"Data date range: {data['date'].min()} to {data['date'].max()}")

# Prepare features
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Prepare X and y
X = data[['feature1', 'feature2', 'day_of_week', 'month', 'is_weekend']]
y = data['sales']

# Train models
linear_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

models = {
    'linear': linear_model,
    'random_forest': rf_model
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=60, query_string=True)
def predict():
    try:
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        date = pd.to_datetime(request.form['date'])
        model_name = request.form.get('model', 'linear')
        
        input_data = [[feature1, feature2, date.dayofweek, date.month, int(date.dayofweek in [5, 6])]]
        
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        
        if model_name == 'random_forest':
            importances = model.feature_importances_
            feature_importance = dict(zip(['feature1', 'feature2', 'day_of_week', 'month', 'is_weekend'], importances))
        else:
            feature_importance = dict(zip(['feature1', 'feature2', 'day_of_week', 'month', 'is_weekend'], model.coef_))
        
        return jsonify({
            'prediction': round(prediction, 2),
            'feature_importance': feature_importance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/historical_data')
def get_historical_data():
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    print(f"Received request for historical data: start_date={start_date}, end_date={end_date}")  # Debug print
    
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        if filtered_data.empty:
            return jsonify({"error": "No data found for the given date range"}), 404
        
        result = filtered_data[['date', 'sales']].to_dict('records')
        
        # Ensure dates are serialized as strings
        for item in result:
            item['date'] = item['date'].strftime('%Y-%m-%d')
        
        print(f"Returning {len(result)} records")  # Debug print
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_historical_data: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    return jsonify({"message": "Test route working"})

if __name__ == '__main__':
    app.run(debug=True)
