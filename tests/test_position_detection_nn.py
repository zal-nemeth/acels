import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from acels.position_detection_nn import (
    convert_model_to_c_source,
    denorm,
    evaluate_regression_model,
    install_xxd,
    norm,
    train_model,
)


def test_norm():
    # Prepare data
    data = np.array([10, 20, 30])
    mean = 20
    std = 5

    # Expected normalized data
    expected_normalized_data = np.array([-2, 0, 2])

    # Normalize
    normalized_data = norm(data, mean, std)

    # Check if the result matches the expected output
    np.testing.assert_array_almost_equal(normalized_data, expected_normalized_data)


def test_denorm():
    # Prepare normalized data
    normalized_data = np.array([-2, 0, 2])
    mean = 20
    std = 5

    # Expected denormalized data
    expected_data = np.array([10, 20, 30])

    # Denormalize
    denormalized_data = denorm(normalized_data, mean, std)

    # Check if the result matches the expected output
    np.testing.assert_array_almost_equal(denormalized_data, expected_data)


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
    assert metrics["MAE"] == pytest.approx(expected_mae, abs=1e-2)
    assert metrics["MSE"] == pytest.approx(expected_mse, abs=1e-2)
    assert metrics["RMSE"] == pytest.approx(expected_rmse, abs=1e-2)
    assert metrics["R²"] == pytest.approx(expected_r_squared, abs=1e-2)


@patch("subprocess.run")
def test_install_xxd(mock_run):
    install_xxd()
    expected_calls = [
        call(["apt-get", "update"], check=True),
        call(["apt-get", "-qq", "install", "xxd"], check=True),
    ]
    mock_run.assert_has_calls(expected_calls, any_order=True)


def test_convert_model_to_c_source():
    model_tflite = "dummy_model.tflite"
    model_tflite_micro = "dummy_model.cc"

    with patch("subprocess.run") as mock_run:
        convert_model_to_c_source(model_tflite, model_tflite_micro)
        mock_run.assert_called_once_with(
            ["xxd", "-i", model_tflite, model_tflite_micro], check=True
        )


@patch(
    "acels.position_detection_nn.tf.keras.models.Sequential"
)  # Mocking TensorFlow Sequential model
@patch("acels.position_detection_nn.pd.read_csv")
def test_train_model(mock_read_csv, mock_sequential):
    dummy_data = pd.DataFrame(
        {
            "s1": np.random.rand(10),
            "s2": np.random.rand(10),
            "s3": np.random.rand(10),
            "s4": np.random.rand(10),
            "s5": np.random.rand(10),
            "s6": np.random.rand(10),
            "s7": np.random.rand(10),
            "s8": np.random.rand(10),
            "x": np.random.rand(10),
            "y": np.random.rand(10),
            "z": np.random.rand(10),
        }
    )
    mock_read_csv.return_value = dummy_data

    mock_model = MagicMock()
    mock_sequential.return_value = mock_model
    # Setup mock return values to simulate training behavior
    mock_model.fit.return_value = MagicMock(history={"loss": [0.5], "mae": [0.2]})
    mock_model.evaluate.return_value = [0.5, 0.2]  # Mock loss and metric values

    # Call the function with mocked dependencies
    train_model(
        model_id="dummy_id",
        training_data="tests/dummy_training_data.csv",
        model_path="tests/dummy_model",
        epochs=10,
        batch_size=32,
    )

    expected_predictions_path = "tests/expected_dummy_predictions.csv"
    expected_df = pd.read_csv(expected_predictions_path)

    og_pred_path = "acels/predictions/dummy_id_og_predictions.csv"
    actual_df = pd.read_csv(og_pred_path)

    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, atol=1e-2)

    # Define the file path
    metrics_path = "acels/metrics/model_dummy_id_og_metrics.txt"
    split_path = "acels/data/dummy_id_split_feature_data.csv"
    test_coord_path = "acels/data/dummy_id_test_coordinates.csv"
    svg_path = "acels/figures/dummy_id_model_eval.svg"
    loss_path = "acels/figures/dummy_id_training_loss_metrics.svg"
    stat_path = "tests/dummy_data_statistics.csv"

    # # Check if the file exists and delete it
    if (
        os.path.exists(metrics_path)
        and os.path.exists(split_path)
        and os.path.exists(test_coord_path)
        and os.path.exists(og_pred_path)
        and os.path.exists(svg_path)
        and os.path.exists(loss_path)
        and os.path.exists(stat_path)
    ):

        os.remove(metrics_path)
        os.remove(split_path)
        os.remove(test_coord_path)
        os.remove(og_pred_path)
        os.remove(svg_path)
        os.remove(stat_path)
        os.remove(loss_path)

    else:
        raise FileNotFoundError("File is missing")
