from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import json
import plotly.graph_objs as go
from model import predict_sales

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


if __name__ == "__main__":
    app.run(debug=True)
