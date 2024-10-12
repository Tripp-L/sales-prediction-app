import subprocess

def run_sales_prediction():
    try:
        result = subprocess.run(['python', 'sales_prediction.py'], 
                                check=True, 
                                capture_output=True, 
                                text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sales_prediction.py: {e}")
        print(e.output)

if __name__ == "__main__":
    print("Running Sales Prediction App...")
    run_sales_prediction()
    print("Sales Prediction App completed.")
