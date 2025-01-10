from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import json
import plotly.graph_objs as go
from model import predict_sales
import pandas as pd
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket connections if needed


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


def generate_visualization(input_data, prediction):
    """Generate a visualization of the input features and the predicted sales."""
    features = ["Traffic", "Marketing", "Advertising", "Social"]
    values = [
        float(input_data["traffic"]),
        float(input_data["marketing"]),
        float(input_data["advertising"]),
        float(input_data["social"]),
    ]

    trace1 = go.Bar(
        x=features,
        y=values,
        name="Input Features",
        marker=dict(color="rgb(26, 118, 255)"),
    )

    trace2 = go.Bar(
        x=["Predicted Sales"],
        y=[float(prediction)],
        name="Prediction",
        marker=dict(color="rgb(55, 83, 109)"),
    )

    data = [trace1, trace2]
    layout = go.Layout(
        title="Sales Prediction Visualization",
        xaxis=dict(title="Features"),
        yaxis=dict(title="Values"),
        barmode="group",
    )

    return json.loads(json.dumps(go.Figure(data=data, layout=layout).to_dict()))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Incoming request data:", data)

        if not data:
            raise ValueError("No data received in the request.")

        required_keys = ["traffic", "marketing", "advertising", "social", "ml_model"]
        for key in required_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Missing or invalid parameter: {key}")

        input_data = {
            "traffic": float(data.get("traffic", 0)),
            "marketing": float(data.get("marketing", 0)),
            "advertising": float(data.get("advertising", 0)),
            "social": float(data.get("social", 0)),
            "ml_model": data.get("ml_model", "random_forest"),
        }

        prediction = predict_sales(input_data)
        visualization = generate_visualization(input_data, prediction)

        response_data = {
            "success": True,
            "prediction": float(prediction),
            "accuracy_score": 0.85,
            "confidence_score": 0.78,
            "plot": visualization,
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }), 400


@app.route("/apply_filters", methods=["POST"])
def apply_filters():
    try:
        data = request.get_json()
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        metrics = data.get('metrics', [])

        # Here you would filter your actual data based on the parameters
        filtered_data = filter_data(start_date, end_date, metrics)
        
        return jsonify({
            "success": True,
            "data": filtered_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/export_data", methods=["POST"])
def export_data():
    try:
        data = request.get_json()
        format_type = data.get('format')
        include_analysis = data.get('includeAnalysis', False)
        
        # Get filtered data
        filtered_data = get_filtered_data(data)
        
        if format_type == 'csv':
            output = BytesIO()
            filtered_data.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='sales_data.csv'
            )
        elif format_type == 'json':
            return jsonify(filtered_data)
        elif format_type == 'excel':
            output = BytesIO()
            filtered_data.to_excel(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='sales_data.xlsx'
            )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/update_visualization", methods=["POST"])
def update_visualization():
    try:
        data = request.get_json()
        viz_type = data.get('type')
        
        # Create base data structure for visualization
        plot_data = {
            'data': [],
            'layout': {
                'title': '',
                'showlegend': True,
                'template': 'plotly_dark'
            }
        }
        
        if viz_type == 'sales_trend':
            # Simple line chart for sales trend
            plot_data['data'] = [{
                'type': 'scatter',
                'mode': 'lines+markers',
                'x': list(range(10)),  # Replace with actual dates
                'y': list(range(10)),  # Replace with actual sales data
                'name': 'Sales Trend'
            }]
            plot_data['layout']['title'] = 'Sales Trend Over Time'
            
        elif viz_type == 'advanced_dashboard':
            # Multiple traces for advanced dashboard
            plot_data['data'] = [
                {
                    'type': 'bar',
                    'x': ['Traffic', 'Marketing', 'Advertising', 'Social'],
                    'y': [100, 200, 150, 300],  # Replace with actual metrics
                    'name': 'Current Period'
                },
                {
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'x': ['Traffic', 'Marketing', 'Advertising', 'Social'],
                    'y': [90, 180, 160, 280],  # Replace with actual historical data
                    'name': 'Previous Period'
                }
            ]
            plot_data['layout']['title'] = 'Advanced Performance Dashboard'
            plot_data['layout']['barmode'] = 'group'
            
        return jsonify({
            "success": True,
            "plot": plot_data
        })
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")  # Add logging
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/get_latest_data")
def get_latest_data():
    try:
        # Fetch and return the latest data
        latest_data = fetch_latest_data()
        return jsonify({
            "success": True,
            "data": latest_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
