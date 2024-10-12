import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import numpy as np

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
file_path = os.path.join(script_dir, 'sales_data.csv')

try:
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    print(f"Successfully loaded data from: {file_path}")
except FileNotFoundError:
    print(f"Error: 'sales_data.csv' not found at {file_path}")
    exit(1)

# Data Cleaning and Preprocessing
print(data.head())
print(data.info())

# Handle missing values if any (only for numeric columns)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['sales'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_plot.png')
plt.close()

# Prepare for Modeling
X = data[['feature1', 'feature2']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Conclusion
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.tight_layout()
plt.savefig('prediction_plot.png')
plt.close()

print("Sales prediction script completed successfully.")

print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir())

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})
print("\nFeature Importance:")
print(feature_importance)

# Make a prediction for the next day
last_day = data.iloc[-1]
next_day_features = [[last_day['feature1'], last_day['feature2']]]
next_day_prediction = model.predict(next_day_features)
print(f"\nPredicted sales for the next day: {next_day_prediction[0]:.2f}")
