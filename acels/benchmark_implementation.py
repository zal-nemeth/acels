import time
import csv

import serial
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression_model(original_data, predicted_data):
    """
    Evaluates a regression model using Mean Absolute Error (MAE), Mean Squared Error (MSE), 
    Root Mean Squared Error (RMSE), and R-squared (R²), alongside a custom accuracy percentage 
    based on the data range.

    Parameters:
    - original_data (numpy.ndarray or pandas.DataFrame): The actual values.
    - predicted_data (numpy.ndarray or pandas.DataFrame): The predicted values by the model.

    Returns:
    - A dictionary containing MAE, MSE, RMSE, R², and custom accuracy percentage.
    """
    if isinstance(original_data, pd.DataFrame):
        original_data = original_data.to_numpy()
    if isinstance(predicted_data, pd.DataFrame):
        predicted_data = predicted_data.to_numpy()
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(original_data, predicted_data)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(original_data, predicted_data)
    
    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    r_squared = r2_score(original_data, predicted_data)
    
    # Estimate the range of the data
    data_range = np.max(original_data) - np.min(original_data)
    
    # Calculate custom accuracy percentage
    accuracy = (1 - rmse / data_range) * 100
    accuracy_percentage = np.clip(accuracy, 0, 100)  # Ensure the percentage is between 0 and 100
    
    # Print and return the evaluation metrics
    evaluation_metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r_squared,
        "Accuracy Percentage": accuracy_percentage
    }
    
    for metric, value in evaluation_metrics.items():
        print(f"# {metric}: {value:.2f}")
    
    return evaluation_metrics

def compare_datasets(original_csv, predicted_csv, existing=True):
    # Load the datasets
    original_data = pd.read_csv(original_csv, usecols=['x', 'y', 'z'])
    predicted_data = pd.read_csv(predicted_csv, usecols=['x', 'y', 'z'])

    # Check if predicted data exists and has more than 5 rows
    if not existing:
        serial_port = 'COM4'
        baud_rate = 9600

        # Initialize serial connection
        ser = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Wait for the serial connection to initialize

        with open(original_csv, mode='r') as infile, open(predicted_csv, mode='w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Skip header row in input, write header to output
            next(reader)
            writer.writerow(['x', 'y', 'z'])  # Assuming you only want x, y, z in the output

            for row in reader:
                # Send s1 through s8 as a comma-separated string, then read the response
                data_string = ','.join(row[:8]) + '\n'  # Only take first 8 columns
                ser.write(data_string.encode())

                # Read the Arduino's response
                response = ser.readline().decode().strip()
                if response:  # If there's a response, write it to the output CSV
                    writer.writerow(response.split(','))

        # Close the serial connection
        ser.close()

    # Evaluate the regression model
    evaluation_metrics = evaluate_regression_model(original_data, predicted_data)
    return evaluation_metrics

#-------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------
original_csv_path = 'acels/test_coordinates.csv'
predicted_csv_path = 'acels/quantized_implementation_output.csv'
metrics = compare_datasets(original_csv_path, predicted_csv_path, False)

print(metrics)