import os
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from acels.benchmark_model_impl import compare_datasets, evaluate_regression_model

# @patch("acels.benchmark_model_impl.pd.read_csv")
# @patch("acels.benchmark_model_impl.evaluate_regression_model")
# def test_compare_datasets(mock_evaluate_regression_model, mock_read_csv):
#     # Setup mock return values
#     mock_evaluate_regression_model.return_value = {
#         "Model ID": "01",
#         # Include mock return values for other metrics as needed
#     }

#     # Assuming `original_csv` and `predicted_csv` contain the same perfect prediction data
#     mock_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
#     mock_read_csv.return_value = mock_df

#     model_id, model_type, data_name = "01", "og", "Test Data"
#     original_csv = "acels/data/00_test_coordinates.csv"
#     predicted_csv = "acels/predictions/00_og_predictions.csv"

#     # Call compare_datasets
#     metrics = compare_datasets(
#         model_id, model_type, original_csv, predicted_csv, existing=True
#     )

#     # Verify evaluate_regression_model was called with expected arguments
#     mock_evaluate_regression_model.assert_called_once_with(
#         model_id, model_type, mock_df, mock_df, data_name
#     )


def test_evaluate_regression_model():
    # Mock data for original and predicted values
    original_data = np.array([3, -0.5, 2, 7])
    predicted_data = np.array([2.5, 0.0, 2, 8])

    # Evaluate the model
    metrics = evaluate_regression_model(
        "test_model", "non-quantized", original_data, predicted_data
    )

    # Expected values for metrics
    expected_mae = 0.5
    expected_mse = 0.375
    expected_rmse = np.sqrt(expected_mse)
    expected_r_squared = 0.9489795918367347

    # Define the file path
    file_path = "acels/metrics/model_test_model_non-quantized_metrics.txt"

    # Check if the file exists and delete it
    if os.path.exists(file_path):
        os.remove(file_path)
        result = "File deleted successfully."
    else:
        result = "File does not exist."

    result

    # Check if the results match the expected output
    assert metrics["MAE"][0] == pytest.approx(expected_mae, abs=1e-2)
    assert metrics["MSE"][0] == pytest.approx(expected_mse, abs=1e-2)
    assert metrics["RMSE"][0] == pytest.approx(expected_rmse, abs=1e-2)
    assert metrics["R²"][0] == pytest.approx(expected_r_squared, abs=1e-2)
