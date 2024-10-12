import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load Your Data
try:
    data = pd.read_csv('sales_data.csv')
except FileNotFoundError:
    print("Error: 'sales_data.csv' not found. Please ensure the file is in the correct directory.")
    exit(1)

# Step 3: Data Cleaning and Preprocessing
print(data.head())
print(data.info())

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Normalize the dataset
scaler = StandardScaler()
data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])

# After loading the data
data['date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Update your feature list
features = ['feature1', 'feature2', 'day_of_week', 'month', 'is_weekend']
X = data[features]
y = data['sales']

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='sales', data=data)
plt.title('Sales Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_over_time.png')
plt.close()

data['month'] = pd.to_datetime(data['date']).dt.month
monthly_sales = data.groupby('month')['sales'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig('monthly_sales.png')
plt.close()

# Step 5: Prepare for Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# After your model training
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.close()

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Step 8: Addressing Challenges
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

print(f'Lasso Mean Absolute Error: {lasso_mae}')
print(f'Lasso R-squared: {lasso_r2}')

# Step 9: Conclusion
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.tight_layout()
plt.savefig('actual_vs_predicted_sales.png')
plt.close()

print("Analysis complete. Check the generated PNG files for visualizations.")
