import csv
import json
import time

import numpy as np
import pandas as pd
import serial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(
    model_id, model_type, original_data, predicted_data, data_name
):
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

    file_name = f"model_{model_id}_{model_type}_metrics.txt"

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

    with open(file_name, "w") as f:
        json.dump(evaluation_metrics, f)

    print(f"{data_name} metrics")
    for metric, value in evaluation_metrics.items():
        print(f"# {metric}: {value[0]:.3f} {value[1]}")

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
model_id = "01"

original_csv_path = f"acels/data/test_coordinates_{model_id}.csv"

# Full model Results
model_type_og = "og"
full_model_pred = f"acels/predictions/full_model_predictions_{model_id}.csv"

# Non-quantized results
model_type_non_quant = "non_quant_impl"
non_quant_pred = f"acels/predictions/non_quantized_predictions_{model_id}.csv"
non_quant_impl_pred = (
    f"acels/predictions/non_quantized_implementation_output_{model_id}.csv"
)

# Quantized results
model_type_quant = "quant_impl"
quant_pred = f"acels/predictions/quantized_predictions_{model_id}.csv"
quant_impl_pred = f"acels/predictions/quantized_implementation_output_{model_id}.csv"

metrics_full_model = compare_datasets(
    model_id, model_type_og, original_csv_path, full_model_pred, "Full model", True
)
metrics_non_quant_pred_impl = compare_datasets(
    model_id,
    model_type_non_quant,
    original_csv_path,
    non_quant_impl_pred,
    "Non-quantized implemented model",
    True,
)
metrics_quant_pred_impl = compare_datasets(
    model_id,
    model_type_non_quant,
    original_csv_path,
    quant_impl_pred,
    "Quantized implemented model",
    True,
)
