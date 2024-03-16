import os
import csv
import json
import time

import numpy as np
import pandas as pd
import serial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(model_id, model_type, original_data, predicted_data):
    """
    Evaluates a regression model using Mean Absolute Error (MAE), Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), and R-squared (R²), alongside a custom accuracy percentage
    based on the data range.

    Parameters:
    - original_data (numpy.ndarray or pandas.DataFrame): The actual values.
    - predicted_data (numpy.ndarray or pandas.DataFrame): The predicted values by the model.
    - model_id (str): ID of the model.
    - model_type (str): Model type i.e. tflite, quantized, non-quantized etc.

    Returns:
    - A dictionary containing MAE, MSE, RMSE, R², and custom accuracy percentage.
    """

    file_name = f"acels/metrics/model_{model_id}_{model_type}_metrics.txt"

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
    accuracy_percentage = np.clip(
        accuracy, 0, 100
    )  # Ensure the percentage is between 0 and 100

    # Print and return the evaluation metrics
    evaluation_metrics = {
        "Model ID": model_id,
        "MAE": (mae, "mm"),
        "MSE": (mse, "mm²"),
        "RMSE": (rmse, "mm"),
        "R²": (r_squared, ""),
        "Accuracy Percentage": (accuracy_percentage, "%"),
    }

    # Check if file exists to append or write new
    mode = "a" if os.path.exists(file_name) else "w"

    with open(file_name, mode) as f:
        for metric, value in evaluation_metrics.items():
            if isinstance(value, str) or value[1] == "%":
                continue
            # Write the formatted string to the file
            f.write(f"# {metric}: {value[0]:.4f} {value[1]}\n")

    print(f"\n{model_type} model metrics")
    for metric, value in evaluation_metrics.items():
        if isinstance(value, str):
            continue
        print(f"# {metric}: {value[0]:.4f} {value[1]}")

    return evaluation_metrics


def compare_datasets(
    model_id, model_type, original_csv, predicted_csv, data_name, existing=True
):
    # Load the datasets
    original_data = pd.read_csv(original_csv, usecols=["x", "y", "z"])
    predicted_data = pd.read_csv(predicted_csv, usecols=["x", "y", "z"])

    # Check if predicted data exists and has more than 5 rows
    if not existing:
        serial_port = "COM4"
        baud_rate = 9600

        # Initialize serial connection
        ser = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Wait for the serial connection to initialize

        with open(original_csv, mode="r") as infile, open(
            predicted_csv, mode="w", newline=""
        ) as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Skip header row in input, write header to output
            next(reader)
            writer.writerow(
                ["x", "y", "z"]
            )  # Assuming you only want x, y, z in the output

            for row in reader:
                # Send s1 through s8 as a comma-separated string, then read the response
                data_string = ",".join(row[:8]) + "\n"  # Only take first 8 columns
                ser.write(data_string.encode())

                # Read the Arduino's response
                response = ser.readline().decode().strip()
                if response:  # If there's a response, write it to the output CSV
                    writer.writerow(response.split(","))

        # Close the serial connection
        ser.close()

    # Evaluate the regression model
    evaluation_metrics = evaluate_regression_model(
        model_id, model_type, original_data, predicted_data, data_name
    )
    return evaluation_metrics


# -------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    model_id = "01"

    original_csv_path = f"acels/data/{model_id}_test_coordinates.csv"

    # Full model Results
    model_type_og = "og"
    full_model_pred = f"acels/predictions/{model_id}_og_predictions.csv"

    # Non-quantized results
    model_type_non_quant = "non_quant_impl"
    non_quant_pred = f"acels/predictions/{model_id}_non_quantized_predictions.csv"
    non_quant_impl_pred = (
        f"acels/predictions/{model_id}_non_quantized_implementation_output.csv"
    )

    # Quantized results
    model_type_quant = "quant_impl"
    quant_pred = f"acels/predictions/{model_id}_quantized_predictions.csv"
    quant_impl_pred = (
        f"acels/predictions/{model_id}_quantized_implementation_output.csv"
    )

    metrics_full_model = compare_datasets(
        model_id, model_type_og, original_csv_path, full_model_pred, True
    )
    metrics_non_quant_pred_impl = compare_datasets(
        model_id, model_type_non_quant, original_csv_path, non_quant_impl_pred, True
    )
    metrics_quant_pred_impl = compare_datasets(
        model_id, model_type_non_quant, original_csv_path, quant_impl_pred, True
    )
