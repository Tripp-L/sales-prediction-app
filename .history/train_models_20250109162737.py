from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import numpy as np
import os

# Ensure the 'models' directory exists
if not os.path.exists("models"):
    os.makedirs("models")
    print("Created 'models' directory.")

# Generate synthetic data for training
np.random.seed(42)
X = np.random.rand(1000, 4) * 1000  # Features: traffic, marketing, advertising, social
y = (
    X[:, 0] * 0.3 + X[:, 1] * 0.25 + X[:, 2] * 0.2 + X[:, 3] * 0.15 + np.random.rand(1000) * 500
)  # Target: a combination of features with some random noise

# Define models to train
models = {
    "random_forest": RandomForestRegressor(),
    "gradient_boost": GradientBoostingRegressor(),
    "xgboost": XGBRegressor(),
    "lightgbm": LGBMRegressor(),
    "svr": SVR(),
    "neural_network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500),
}

# Train each model and save it
for name, model in models.items():
    try:
        print(f"Training {name}...")
        model.fit(X, y)  # Train the model
        model_path = f"models/{name}.joblib"
        joblib.dump(model, model_path)  # Save the model
        print(f"{name} model saved to {model_path}.")
    except Exception as e:
        print(f"Error training {name}: {e}")

print("Model training complete.")
