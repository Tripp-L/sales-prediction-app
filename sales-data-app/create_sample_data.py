import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
np.random.seed(42)
sales = np.random.randint(800, 1500, size=len(dates))
feature1 = np.random.randint(1, 10, size=len(dates))
feature2 = np.random.randint(1, 5, size=len(dates))

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'feature1': feature1,
    'feature2': feature2
})

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print("Sample data created and saved to 'sales_data.csv'")
