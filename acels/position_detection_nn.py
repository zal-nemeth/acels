# -------------------------------------------------------------------------------------------------
# Import libraries
import argparse
import csv
import io
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense

from acels.benchmark_implementation import evaluate_regression_model


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Define functions
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Normalize data
def norm(x, mean, std):
    """
    Normalizes a pandas Series or a numpy array using pre-calculated mean and
    standard deviation from training statistics.

    The function expects `x` to be either a pandas Series or a numpy array representing
    a single feature or multiple features.

    Parameters:
    - x (pandas.Series or numpy.ndarray): The data to be normalized.

    Returns:
    - pandas.Series or numpy.ndarray: The normalized data.
    """
    normed_value = (x - mean) / std
    return normed_value


def denorm(x, mean, std):
    """
    Denormalizes a value or array of values using pre-calculated mean and standard deviation.

    This function is intended to reverse the normalization process for a dataset.
    Parameters:
    - x (float or numpy.ndarray or pandas.Series): The normalized value(s) to be denormalized.
        This can be a single floating-point number, a NumPy array, or a pandas Series.

    Returns:
    - denormed_value (float or numpy.ndarray or pandas.Series): The denormalized value(s).
        The return type matches the input type.
    """
    denormed_value = (x * std) + mean
    return denormed_value


def install_xxd():
    """
    Installs the xxd utility if it's not already available on the system.
    """
    try:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "xxd"], check=True)
        print("xxd installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install xxd: {e}")


def convert_model_to_c_source(model_tflite, model_tflite_micro):
    """
    Converts a TensorFlow Lite model to a C source file format.

    Parameters:
    - model_tflite (str): Path to the TensorFlow Lite model file.
    - model_tflite_micro (str): Output path for the C source file.
    """
    try:
        # Convert model to a C source file
        subprocess.run(["xxd", "-i", model_tflite, model_tflite_micro], check=True)
        print(f"Model converted to C source: {model_tflite_micro}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert model: {e}")


def update_variable_names(model_tflite_micro, original_name, new_name="position_model"):
    """
    Updates variable names in the generated C source file.

    Parameters:
    - model_tflite_micro (str): Path to the C source file.
    - original_name (str): Original variable name to replace.
    - new_name (str): New variable name.
    """
    try:
        replace_text = original_name.replace("/", "_").replace(".", "_")
        sed_cmd = f"s/{replace_text}/{new_name}/g"
        subprocess.run(["sed", "-i", sed_cmd, model_tflite_micro], check=True)
        print(f"Variable names updated in {model_tflite_micro}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update variable names: {e}")


def data_processing(input_data):
    """
    Processes input data by randomizing its order and converting its type.

    Parameters:
    - input_data (str): The file path to the dataset to be processed.

    Returns:
    - data32 (DataFrame): The processed data with randomized order and converted to float32.
    - train_stats (DataFrame): Basic statistics of the processed data, with each row representing a different statistic and each column representing a feature of the data.

    Example:
        data, stats = data_processing('path/to/dataset.csv')
    """
    # -----------------------------------------------------------------------------
    # Data Processing
    # -----------------------------------------------------------------------------

    data = pd.read_csv(input_data)

    # num_rows = data.shape[0]
    # Check datatype
    data = data.sample(frac=1).reset_index(drop=True)
    # Convert dataframe to float 32
    data32 = data.astype(np.float32)

    # Obtain data statistics
    train_stats = data32.describe()
    train_stats = train_stats.transpose()

    return data32, train_stats


# -----------------------------------------------------------------------------
# Model definition and training
# -----------------------------------------------------------------------------
def train_model(model_id, training_data, model_path, epochs=1000, batch_size=32):
    """
    Trains a neural network model on provided dataset and evaluates its performance.

    ### Parameters:
    - model_id (str): Identifier for the model, used for naming output files.
    - training_data (str): File path to the dataset to be used for training.
    - model_path (str): Path where the trained model will be saved.
    - epochs (int, optional): Number of epochs to train the model. Default is 1000.
    - batch_size (int, optional): Batch size to use during training. Default is 32.

    ### Returns:
    - None. This function saves the trained model, metrics, and plots to the specified paths.

    ### Example:
    - train_model('model1', 'path/to/training_data.csv', 'path/to/save/model', epochs=100, batch_size=32) ->

    Trains a model with ID 'model1' using data from 'path/to/training_data.csv', saves it to 'path/to/save/model', and runs for 100 epochs with a batch size of 32.

    ### Note:
    The input dataset is expected to be in CSV format and contain columns named 's1' to 's8' for features and 'x', 'y', 'z' for targets.
    The function automatically randomizes the dataset, normalizes it using statistics computed from the training set,
    and splits it into training (60%), validation (20%), and test (20%) sets.
    It saves various output files, including model details, training and validation metrics,
    and plots to visualize training progress and model performance, in directories specified by the model_id and model type.
    """
    model_type = "og"
    data32, train_stats = data_processing(training_data)

    # Print and save to csv for reuse in control software
    print(f"\nData Statistics: {train_stats}")
    if "dummy" in model_id:
        train_stats.to_csv("tests/dummy_data_statistics.csv", index=True)
    else:
        train_stats.to_csv("acels/data/data_statistics.csv", index=True)

    # Separate Data into Feature and Target Variables
    # The `_og` suffix refers to the original data without normalization
    # It is assign to a variable to be later used for testing purposes
    feature_data_og = data32[["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]]
    target_data_og = data32[["x", "y", "z"]]

    # Split the data into  training and test sections
    TRAIN_SPLIT = int(0.6 * feature_data_og.shape[0])
    TEST_SPLIT = int(0.2 * feature_data_og.shape[0] + TRAIN_SPLIT)

    _, feature_test_og, _ = np.split(feature_data_og, [TRAIN_SPLIT, TEST_SPLIT])
    _, target_test_og, _ = np.split(target_data_og, [TRAIN_SPLIT, TEST_SPLIT])

    mean = train_stats["mean"][:].values
    std = train_stats["std"][:].values

    # Normalize data
    normed_data = norm(data32, mean, std)
    normed_data.head()

    # -----------------------------------------------------------------------------
    # Data Splitting
    # -----------------------------------------------------------------------------
    # Separate Data into Feature and Target Variables
    feature_data = normed_data[["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]]
    target_data = normed_data[["x", "y", "z"]]

    feature = feature_data
    target = target_data

    # Assign 60% of data for training
    # Assign 20% of data for testing
    # Assign 20% pf data to validation
    TRAIN_SPLIT = int(0.6 * feature.shape[0])
    TEST_SPLIT = int(0.2 * feature.shape[0] + TRAIN_SPLIT)

    feature_train, feature_test, feature_validate = np.split(
        feature, [TRAIN_SPLIT, TEST_SPLIT]
    )
    target_train, target_test, target_validate = np.split(
        target, [TRAIN_SPLIT, TEST_SPLIT]
    )

    feature_train.to_csv(f"acels/data/{model_id}_split_feature_data.csv", index=False)

    # -----------------------------------------------------------------------------
    # Model Building
    # -----------------------------------------------------------------------------
    # Create model with 8 input, 3 output and 5 hidden layers
    model = tf.keras.Sequential()
    model.add(Dense(60, activation="tanh", input_shape=(8,)))
    model.add(Dense(80, activation="tanh"))
    model.add(Dense(80, activation="tanh"))
    model.add(Dense(60, activation="tanh"))
    model.add(Dense(30, activation="tanh"))
    model.add(Dense(3))
    model.compile(optimizer="nadam", loss="mse", metrics=["mae"])
    model.summary()

    # -----------------------------------------------------------------------------
    # Model Training
    # -----------------------------------------------------------------------------
    # Train model
    history_1 = model.fit(
        feature_train,
        target_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(feature_validate, target_validate),
    )
    # Check Mean Absolute Error
    _, test_mae = model.evaluate(feature_test, target_test, verbose=0)
    print("Testing set Mean Abs Error: {:5.3f} mm".format(test_mae))

    # Save model to disk
    model.save(model_path)

    # -----------------------------------------------------------------------------
    # Save model details
    # -----------------------------------------------------------------------------
    # Retrieving model details
    optimizer_name = (
        model.optimizer._name
        if hasattr(model.optimizer, "_name")
        else model.optimizer.__class__.__name__
    )
    loss = model.loss if hasattr(model, "loss") else "Loss information not available"
    metrics = (
        [m.name for m in model.metrics]
        if hasattr(model, "metrics")
        else "Metrics information not available"
    )

    # Prepare to capture the model's summary
    str_io = io.StringIO()
    model.summary(print_fn=lambda x: str_io.write(x + "\n"))
    model_summary = str_io.getvalue()

    # Capturing layer details, specifically activation functions
    layer_details = ""
    for layer in model.layers:
        config = layer.get_config()
        activation = config.get("activation", "None")
        layer_details += f"Layer: {layer.name}, Activation: {activation}\n"

    # Full details to write
    full_details = f"""Model ID: {model_id}\n\n{model_summary}\n{
        layer_details}\nOptimizer: {optimizer_name}\nLoss: {loss}\nMetrics: {str(metrics)}\n"""

    # Additional details
    additional_info = f"\nEpochs: {epochs}\nBatch Size: {batch_size}\n\n"

    # Writing to a text file
    metrics_file_name = f"acels/metrics/{model_id}_model_{model_type}_metrics.txt"
    with open(metrics_file_name, "w") as file:
        file.write(full_details)
        file.write(additional_info)

    # -----------------------------------------------------------------------------
    # Plot Training Metrics
    # -----------------------------------------------------------------------------
    train_loss = history_1.history["loss"]
    val_loss = history_1.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)
    train_mae = history_1.history["mae"]
    val_mae = history_1.history["val_mae"]
    SKIP = 50
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 plot

    # Plot Training and Validation Loss
    axs[0, 0].plot(epochs, train_loss, "g.", label="Training loss")
    axs[0, 0].plot(epochs, val_loss, "b", label="Validation loss")
    axs[0, 0].set_title("Training and Validation Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Re-plot Training and Validation Loss, skipping first 50
    axs[0, 1].plot(epochs[SKIP:], train_loss[SKIP:], "g.", label="Training loss")
    axs[0, 1].plot(epochs[SKIP:], val_loss[SKIP:], "b.", label="Validation loss")
    axs[0, 1].set_title("Training and Validation Loss (Skip first 50)")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()

    # Plot Training and Validation MAE, skipping first 50
    axs[1, 0].plot(epochs[SKIP:], train_mae[SKIP:], "g.", label="Training MAE")
    axs[1, 0].plot(epochs[SKIP:], val_mae[SKIP:], "b.", label="Validation MAE")
    axs[1, 0].set_title("Training and Validation MAE (Skip first 50)")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("MAE")
    axs[1, 0].legend()

    # Empty plot for symmetry or additional plots if needed
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"acels/figures/{model_id}_training_loss_metrics.svg")
    # plt.show()

    # -----------------------------------------------------------------------------
    # Evaluate Model Predictions
    # -----------------------------------------------------------------------------
    # Calculate and print the loss on our test dataset
    _, test_mae = model.evaluate(feature_test, target_test)
    # Make predictions based on our test dataset
    target_test_pred = model.predict(feature_test)

    # scatter3D requires x, y, and z to be one-dimensional arrays
    norm_x = target_test.iloc[:, 0]
    norm_y = target_test.iloc[:, 1]
    norm_z = target_test.iloc[:, 2]

    norm_x2 = target_test_pred[:, 0]
    norm_y2 = target_test_pred[:, 1]
    norm_z2 = target_test_pred[:, 2]

    # Check model output values
    # Convert to dataframe for denormalization
    pred_df = pd.DataFrame(target_test_pred, columns=["x", "y", "z"])

    # denormed_target = denorm(target)
    denorm_data = denorm(pred_df, mean[8:], std[8:])

    actual_coordinates = target_test_og
    # actual_coordinates_df = pd.DataFrame(actual_coordinates)
    pred_coordinates = denorm_data[["x", "y", "z"]]

    total_test_data = pd.concat([feature_test_og, target_test_og], axis=1)

    total_test_data.to_csv(f"acels/data/{model_id}_test_coordinates.csv", index=False)
    pred_coordinates.to_csv(
        f"acels/predictions/{model_id}_og_predictions.csv", index=False
    )

    x = actual_coordinates.iloc[:, 0]
    y = actual_coordinates.iloc[:, 1]
    z = actual_coordinates.iloc[:, 2]
    x2 = pred_coordinates.iloc[:, 0]
    y2 = pred_coordinates.iloc[:, 1]
    z2 = pred_coordinates.iloc[:, 2]

    eval_metrics_normed = evaluate_regression_model(
        model_id, f"Normalized_{model_type}", target_test, target_test_pred
    )
    # Remove unnecessary metrics file
    if os.path.exists(f"acels/metrics/{model_id}_model_Normalized_og_metrics.txt"):
        os.remove(f"acels/metrics/{model_id}_model_Normalized_og_metrics.txt")
    eval_metrics_og = evaluate_regression_model(
        model_id, model_type, target_test_og, pred_coordinates
    )

    # model_accuracy_normed = eval_metrics_normed["Accuracy Percentage"]
    model_mae_normed = eval_metrics_normed["MAE"][0]
    model_mse_normed = eval_metrics_normed["MSE"][0]
    model_rmse_normed = eval_metrics_normed["RMSE"][0]
    model_r2_normed = eval_metrics_normed["R²"][0]
    # model_accuracy = eval_metrics_og["Accuracy Percentage"]
    model_mae = eval_metrics_og["MAE"][0]
    model_mse = eval_metrics_og["MSE"][0]
    model_rmse = eval_metrics_og["RMSE"][0]
    model_r2 = eval_metrics_og["R²"][0]

    # Adjusting titles, legends, and positioning of the accuracy text to improve clarity and avoid overlay
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0.1},
    )

    # Normalized model predictions plot
    axs[0].scatter3D(
        norm_x, norm_y, norm_z, marker="x", c="blue", s=15, label="Actual Values"
    )
    axs[0].scatter3D(
        norm_x2, norm_y2, norm_z2, c="red", s=8, alpha=0.5, label="Model Predictions"
    )
    axs[0].set_title("Normalized Model Predictions")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[0].legend()

    # Denormalized model predictions plot
    axs[1].scatter3D(x, y, z, c="blue", s=16, label="Actual Values")
    axs[1].scatter3D(x2, y2, z2, c="red", s=8, alpha=0.5, label="Model Predictions")
    axs[1].set_title("Denormalized Model Predictions")
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Y (mm)")
    axs[1].set_zlabel("Z (mm)")
    axs[1].legend()

    # Adjusting text position closer to the plots and using a consistent formatting style for both metrics
    plt.subplots_adjust(bottom=0.15)
    fig.text(
        0.33,
        0.05,
        f"MAE: {model_mae_normed:.3f}, MSE: {model_mse_normed:.3f}, RMSE: {model_rmse_normed:.3f}, R²: {model_r2_normed:.3f}",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.73,
        0.05,
        f"MAE: {model_mae:.3f} mm, MSE: {model_mse:.3f} mm², RMSE: {model_rmse:.3f} mm, R²: {model_r2:.3f}",
        ha="center",
        fontsize=12,
    )

    plt.savefig(f"acels/figures/{model_id}_model_eval.svg")
    # plt.show()


# -----------------------------------------------------------------------------
# Run TensorFlow Lite model
# -----------------------------------------------------------------------------
def run_lite_model(
    test_data_path,
    quant_model_path,
    non_quant_model_path,
    quant_output_path,
    non_quant_output_path,
):
    """
    Executes predictions using both quantized and non-quantized TensorFlow Lite models.

    This function loads test data and performs predictions using both a quantized and a non-quantized
    TensorFlow Lite model. It processes the input features by normalizing them and then converts the
    normalized features to a format suitable for each model type. After prediction, it denormalizes
    the outputs back to their original scale. Finally, the predictions from both models are saved to
    specified output paths in CSV format.

    ### Parameters:
    - test_data_path (str): The path to the test dataset CSV file.
    - quant_model_path (str): The file path to the quantized TensorFlow Lite model.
    - non_quant_model_path (str): The file path to the non-quantized TensorFlow Lite model.
    - quant_output_path (str): The file path where the quantized model predictions will be saved.
    - non_quant_output_path (str): The file path where the non-quantized model predictions will be saved.

    ### Note:
    - The function assumes the first 8 columns of the test data are the input features (s1 to s8) and
    uses global normalization parameters (mean and standard deviation) for preprocessing.
    - Output CSV files contain columns ['x', 'y', 'z'] representing the model's predictions.
    """
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    train_stats = pd.read_csv("acels/data/data_statistics.csv")

    # Separate features and targets
    features = test_data.iloc[:, :8].values  # s1 to s8

    mean = train_stats["mean"][:8].values
    std = train_stats["std"][:8].values
    coord_mean = train_stats["mean"][8:].values
    coord_std = train_stats["std"][8:].values

    norm_features = norm(features, mean, std)
    norm_features32 = norm_features.astype(np.float32)

    # --- Quantized Model Prediction ---
    interpreter_quant = tf.lite.Interpreter(model_path=quant_model_path)
    interpreter_quant.allocate_tensors()

    input_details_quant = interpreter_quant.get_input_details()
    output_details_quant = interpreter_quant.get_output_details()

    # Get the scale and zero_point for input quantization
    input_scale, input_zero_point = input_details_quant[0]["quantization"]

    predictions_quant = []

    for input_data in norm_features32:
        # Quantize the input data
        input_data_quantized = np.round(
            input_data / input_scale + input_zero_point
        ).astype(input_details_quant[0]["dtype"])

        interpreter_quant.set_tensor(
            input_details_quant[0]["index"], [input_data_quantized]
        )
        interpreter_quant.invoke()
        output_data = interpreter_quant.get_tensor(output_details_quant[0]["index"])[0]

        # Dequantize the output data if needed, similar to the input quantization step but in reverse
        output_scale, output_zero_point = output_details_quant[0]["quantization"]
        output_data_dequantized = (output_data - output_zero_point) * output_scale

        predictions_quant.append(output_data_dequantized)

    denorm_predictions_quant = denorm(predictions_quant, coord_mean, coord_std)

    # Save quantized model predictions to CSV
    with open(quant_output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for prediction in denorm_predictions_quant:
            writer.writerow(prediction)

    # TensorFlow Lite Interpreter for Non-Quantized Model
    interpreter_non_quant = tf.lite.Interpreter(model_path=non_quant_model_path)
    interpreter_non_quant.allocate_tensors()
    input_details_non_quant = interpreter_non_quant.get_input_details()
    output_details_non_quant = interpreter_non_quant.get_output_details()

    predictions_non_quant = []
    for input_data in norm_features32:
        interpreter_non_quant.set_tensor(
            input_details_non_quant[0]["index"], [input_data]
        )
        interpreter_non_quant.invoke()
        output_data = interpreter_non_quant.get_tensor(
            output_details_non_quant[0]["index"]
        )[0]
        predictions_non_quant.append(output_data)

    denorm_predictions_non_quant = denorm(predictions_non_quant, coord_mean, coord_std)

    # Save non-quantized model predictions to CSV
    with open(non_quant_output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for prediction in denorm_predictions_non_quant:
            writer.writerow(prediction)


# -----------------------------------------------------------------------------
# Model Conversion
# -----------------------------------------------------------------------------
def convert_model(
    model_id,
    saved_model_path,
    conversion_output_path,
    conversion_output_path_micro,
    conversion_output_path_no_quant,
    conversion_output_path_no_quant_micro,
):
    """
    Converts a TensorFlow saved model to TensorFlow Lite format, both quantized and non-quantized,
    and converts them to C source files for use in microcontroller applications.

    This function takes a TensorFlow model saved in the SavedModel format, converts it to both quantized
    and non-quantized TensorFlow Lite formats, and saves these models to disk. It also includes an option
    to convert these TensorFlow Lite models into C source files suitable for embedding in microcontroller
    applications, using the `xxd` tool for hex dumping and custom functions for conversion and variable renaming.

    ### Parameters:
    - model_id (str): Identifier for the model, used to retrieve specific training data for quantization.
    - saved_model_path (str): File path to the TensorFlow SavedModel to be converted.
    - conversion_output_path (str): File path where the quantized TensorFlow Lite model will be saved.
    - conversion_output_path_micro (str): File path where the C source file for the quantized model will be saved.
    - conversion_output_path_no_quant (str): File path where the non-quantized TensorFlow Lite model will be saved.
    - conversion_output_path_no_quant_micro (str): File path where the C source file for the non-quantized model will be saved.

    ### Note:
    - The function assumes that a representative dataset is available and relevant for the quantization process.
    """
    feature_train = pd.read_csv(f"acels/data/{model_id}_split_feature_data.csv")
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    model_no_quant_tflite = converter.convert()
    # Save the model to disk
    open(conversion_output_path_no_quant, "wb").write(model_no_quant_tflite)

    install_xxd()
    convert_model_to_c_source(
        conversion_output_path_no_quant, conversion_output_path_no_quant_micro
    )
    update_variable_names(
        conversion_output_path_no_quant_micro, conversion_output_path_no_quant
    )

    # Convert the model to the TensorFlow Lite format with quantization
    def representative_dataset():
        for _ in range(500):
            yield ([feature_train.astype(np.float32)])

    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
    model_tflite = converter.convert()

    # Save the model to disk
    open(conversion_output_path, "wb").write(model_tflite)

    install_xxd()
    convert_model_to_c_source(conversion_output_path, conversion_output_path_micro)
    update_variable_names(conversion_output_path_micro, conversion_output_path)


# -----------------------------------------------------------------------------
# Help/Description Function
# -----------------------------------------------------------------------------
def show_help():
    help_message = """
    Usage: python script_name.py [OPTIONS]
    
    Options:
    -t, --train                         Start training a new model. Requires --training_data and --model_path.
        --training_data PATH            Path to the training data.
        --model_path PATH               Path where the trained model will be saved.
        --epochs EPOCHS                 Number of epochs (optional, default=1000).
        --batch_size BATCH_SIZE         Batch size (optional, default=32).

    -r, --read                          Read and execute an existing model. Requires --test_data_path, --model_path, and --output_path.
        --test_data_path PATH           Path to test data (optional, default="acels/test_coordinates.csv").
        --model_path PATH               Path to model (optional, default="acels/models/model.tflite").
        --output_path PATH              Output path for predictions (optional, default="acels/quantized_predictions.csv").

    -c, --convert                       Convert model for embedded use. Requires --saved_model_path and --conversion_output_path.
        --saved_model_path PATH         Path to saved model (optional, default="acels/models/model").
        --conversion_output_path PATH   Output path for the converted model (optional, default="acels/models/model.tflite").

    -h, --help                          Show this help message and exit.
    """
    print(help_message)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Operations", add_help=False)
    parser.add_argument(
        "-t", "--train", action="store_true", help="Start training a new model"
    )
    parser.add_argument(
        "-r", "--read", action="store_true", help="Read and execute an existing model"
    )
    parser.add_argument(
        "-c", "--convert", action="store_true", help="Convert model for embedded use"
    )

    parser.add_argument(
        "-h", "--help", action="store_true", help="Show help message and exit."
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    # Path and variable definitions
    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    MODELS_DIR = "acels/models/"
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    ###########################
    #    Define Parameters    #
    ###########################
    model_id = "01"
    epochs = 1000
    batch_size = 32
    ###########################

    MODEL_TF = MODELS_DIR + "model"
    MODEL_NO_QUANT_TFLITE = MODELS_DIR + f"{model_id}_model_no_quant.tflite"
    MODEL_TFLITE = MODELS_DIR + f"{model_id}_model.tflite"
    MODEL_NO_QUANT_TFLITE_MICRO = MODELS_DIR + f"{model_id}_model_no_quant.cc"
    MODEL_TFLITE_MICRO = MODELS_DIR + f"{model_id}_model.cc"

    training_data = "acels/data/position_data_float_xyz_extended.csv"
    test_data = f"acels/data/{model_id}_test_coordinates.csv"
    quantized_output_path = f"acels/predictions/{model_id}_quantized_predictions.csv"
    non_quantized_output_path = (
        f"acels/predictions/{model_id}_non_quantized_predictions.csv"
    )

    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    # Check if no arguments or the help option was provided

    # Interactive prompt if no arguments provided
    if len(sys.argv) == 1:
        print("\nNo options selected. Choose an operation to perform:\n")
        print("1. Train a new model (-t)")
        print("2. Read and evaluate an existing model (-r)")
        print("3. Convert model for embedded use (-c)")
        print("4. Show help (-h)")
        choice = input("\nEnter the number of your choice: ")

        if choice == "1":
            args.train = True
        elif choice == "2":
            args.read
        elif choice == "3":
            args.convert = True
        elif choice == "4":
            show_help()
            exit()

    # Handle arguments as before...
    if args.help:
        show_help()
    elif args.train:
        train_model(
            model_id=model_id,
            training_data=training_data,
            model_path=MODEL_TF,
            epochs=epochs,
            batch_size=batch_size,
        )

    elif args.read:
        run_lite_model(
            test_data_path=test_data,
            quant_model_path=MODEL_TFLITE,
            non_quant_model_path=MODEL_NO_QUANT_TFLITE,
            quant_output_path=quantized_output_path,
            non_quant_output_path=non_quantized_output_path,
        )

    elif args.convert:
        convert_model(
            model_id,
            MODEL_TF,
            MODEL_TFLITE,
            MODEL_TFLITE_MICRO,
            MODEL_NO_QUANT_TFLITE,
            MODEL_NO_QUANT_TFLITE_MICRO,
        )
