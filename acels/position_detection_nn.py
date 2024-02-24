
# Import libraries
import os
import time
import subprocess

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense


# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

# Normalize data
def norm(x):
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
    normed_value = (x - train_stats['mean']) / train_stats['std']
    return normed_value

def denorm(x):
    """
    Denormalizes a value or array of values using pre-calculated mean and standard deviation.

    This function is intended to reverse the normalization process for a dataset.
    Parameters:
    - x (float or numpy.ndarray or pandas.Series): The normalized value(s) to be denormalized. This can be a single floating-point number, a NumPy array, or a pandas Series.

    Returns:
    - denormed_value (float or numpy.ndarray or pandas.Series): The denormalized value(s). The return type matches the input type.
    """
    denormed_value = (x * train_stats['std']) + train_stats['mean']
    return denormed_value


def evaluate_regression_model(original_data, predicted_data):
    """
    Evaluates a regression model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    and a custom accuracy percentage based on the data range.

    Parameters:
    - original_data (numpy.ndarray or pandas.DataFrame): The actual values.
    - predicted_data (numpy.ndarray or pandas.DataFrame): The predicted values by the model.

    Returns:
    - A dictionary containing MSE, RMSE, and custom accuracy percentage.
    """
    if isinstance(original_data, pd.DataFrame):
        original_data = original_data.to_numpy()
    if isinstance(predicted_data, pd.DataFrame):
        predicted_data = predicted_data.to_numpy()
    
    # Calculate Mean Squared Error
    mse = np.mean((original_data - predicted_data) ** 2)
    
    # Convert to Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Estimate the range of the data
    data_range = np.max(original_data) - np.min(original_data)
    
    # Calculate custom accuracy percentage
    accuracy = (1 - rmse / data_range) * 100
    accuracy_percentage = np.clip(accuracy, 0, 100)  # Ensure the percentage is between 0 and 100
    
    # Print and return the evaluation metrics
    evaluation_metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy Percentage": accuracy_percentage
    }
    
    for metric, value in evaluation_metrics.items():
        print(f"# {metric}: {value:.2f}")
    
    return evaluation_metrics

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
        replace_text = original_name.replace('/', '_').replace('.', '_')
        sed_cmd = f"s/{replace_text}/{new_name}/g"
        subprocess.run(["sed", "-i", sed_cmd, model_tflite_micro], check=True)
        print(f"Variable names updated in {model_tflite_micro}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update variable names: {e}")

# -----------------------------------------------------------------------------
# Define paths to model files
# -----------------------------------------------------------------------------
MODELS_DIR = 'acels/models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
# Assign dataset to data variable
data = pd.read_csv('acels/position_data_float_xyz_extended.csv')

num_rows = data.shape[0]
# Check datatype
data = data.sample(frac=1).reset_index(drop=True)
# Convert dataframe to float 32
data32 = data.astype(np.float32)

# Obtain data statistics
train_stats = data32.describe()
train_stats = train_stats.transpose()

print(f"\nData Statistics: {train_stats}")

# Separate Data into Feature and Target Variables
# The `_og` suffix refers to the original data without normalization
# It is assign to a variable to be later used for testing purposes
feature_data_og = data32[['s1','s2','s3','s4','s5','s6','s7','s8']]
target_data_og = data32[['x', 'y', 'z']]

# Split the data into  training and test sections
TRAIN_SPLIT = int(0.6 * feature_data_og.shape[0])
TEST_SPLIT = int(0.2 * feature_data_og.shape[0] + TRAIN_SPLIT)

feature_train_og, feature_test_og, feature_validate_og = np.split(feature_data_og, [TRAIN_SPLIT, TEST_SPLIT])
target_train_og, target_test_og, target_validate_og = np.split(target_data_og, [TRAIN_SPLIT, TEST_SPLIT])

# Normalize data
normed_data = norm(data32)
normed_data.head()

# -----------------------------------------------------------------------------
# Data Splitting
# -----------------------------------------------------------------------------
# Separate Data into Feature and Target Variables
feature_data = normed_data[['s1','s2','s3','s4','s5','s6','s7','s8']]
target_data = normed_data[['x', 'y', 'z']]

feature = feature_data
target = target_data

# Assign 60% of data for training
# Assign 20% of data for testing
# Assign 20% pf data to validation
TRAIN_SPLIT = int(0.6 * feature.shape[0])
TEST_SPLIT = int(0.2 * feature.shape[0] + TRAIN_SPLIT)

feature_train, feature_test, feature_validate = np.split(feature, [TRAIN_SPLIT, TEST_SPLIT])
target_train, target_test, target_validate = np.split(target, [TRAIN_SPLIT, TEST_SPLIT])

# -----------------------------------------------------------------------------
# Model Building
# -----------------------------------------------------------------------------
# Create model with 8 input, 3 output and 5 hidden layers
model = tf.keras.Sequential()
model.add(Dense(60, activation='tanh', input_shape=(8,)))
model.add(Dense(80, activation='tanh'))
model.add(Dense(80, activation='tanh'))
model.add(Dense(60, activation='tanh'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(3))
model.compile(optimizer='nadam', loss='mse', metrics=['mae'])
model.summary()

# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------
# Train model
history_1 = model.fit(feature_train, target_train, epochs=2000, batch_size=64, validation_data=(feature_validate, target_validate))
# Check Mean Absolute Error
test_loss, test_mae = model.evaluate(feature_test, target_test, verbose=0) 
print('Testing set Mean Abs Error: {:5.3f} mm'.format(test_mae))

# Save model to disk
model.save(MODEL_TF)

# -----------------------------------------------------------------------------
# Plot Training Metrics
# -----------------------------------------------------------------------------
train_loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs = range(1, len(train_loss) + 1)
train_mae = history_1.history['mae']
val_mae = history_1.history['val_mae']
SKIP = 50
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 plot

# Plot Training and Validation Loss
axs[0, 0].plot(epochs, train_loss, 'g.', label='Training loss')
axs[0, 0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0, 0].set_title('Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# Re-plot Training and Validation Loss, skipping first 50
axs[0, 1].plot(epochs[SKIP:], train_loss[SKIP:], 'g.', label='Training loss')
axs[0, 1].plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
axs[0, 1].set_title('Training and Validation Loss (Skip first 50)')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Plot Training and Validation MAE, skipping first 50
axs[1, 0].plot(epochs[SKIP:], train_mae[SKIP:], 'g.', label='Training MAE')
axs[1, 0].plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
axs[1, 0].set_title('Training and Validation MAE (Skip first 50)')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('MAE')
axs[1, 0].legend()

# Empty plot for symmetry or additional plots if needed
axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig("acels/figures/training_loss_metrics.svg")
plt.show()


# -----------------------------------------------------------------------------
# Evaluate Model Predictions
# -----------------------------------------------------------------------------
# Calculate and print the loss on our test dataset
test_loss, test_mae = model.evaluate(feature_test, target_test)
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
pred_df = pd.DataFrame(target_test_pred, columns = ['x','y','z'])

denormed_data = denorm(pred_df)
denormed_feature = denorm(feature)
denormed_target = denorm(target)

df_sensors = denormed_feature[['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']]
df_coordinates = denormed_target[['x', 'y', 'z']]
coordinates2 = denormed_data[['x', 'y', 'z']]

x = df_coordinates.iloc[:, 0]
y = df_coordinates.iloc[:, 1]
z = df_coordinates.iloc[:, 2]
x2 = coordinates2.iloc[:, 0]
y2 = coordinates2.iloc[:, 1]
z2 = coordinates2.iloc[:, 2]

eval_metrics_normed = evaluate_regression_model(target_test, target_test_pred)
eval_metrics_og = evaluate_regression_model(target_test_og, coordinates2)
model_accuracy_normed = eval_metrics_normed["Accuracy Percentage"]
model_accuracy = eval_metrics_og["Accuracy Percentage"]

# Adjusting titles, legends, and positioning of the accuracy text to improve clarity and avoid overlay
fig, axs = plt.subplots(1, 2, figsize=(20, 7), subplot_kw={'projection': '3d'})

# Normalized model predictions plot
axs[0].scatter3D(norm_x, norm_y, norm_z, c='blue', s=15, label='Actual Values')
axs[0].scatter3D(norm_x2, norm_y2, norm_z2, c='red', s=8, alpha=0.5, label='Model Predictions')
axs[0].set_title('Normalized Model Predictions')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_zlabel('Z')
axs[0].legend()

# Denormalized model predictions plot
axs[1].scatter3D(x, y, z, c='blue', s=15, label='Actual Values')
axs[1].scatter3D(x2, y2, z2, c='red', s=8, alpha=0.5, label='Model Predictions')
axs[1].set_title('Denormalized Model Predictions')
axs[1].set_xlabel('X (mm)')
axs[1].set_ylabel('Y (mm)')
axs[1].set_zlabel('Z (mm)')
axs[1].legend()

# Repositioning text for MAE and accuracy to avoid overlay
plt.subplots_adjust(bottom=0.2)
fig.text(0.27,
         0.1,
         f'Normalized Model\nMAE: {test_mae:.3f} mm\nAccuracy: {model_accuracy_normed:.2f}%',
         ha='center',
         fontsize=12)
fig.text(0.73,
         0.1,
         f'Denormalized Model\nMAE: {test_mae:.3f} mm (estimated)\nAccuracy: {model_accuracy:.2f}% (estimated)',
         ha='center',
         fontsize=12)

plt.tight_layout()
plt.savefig("acels/figures/model_eval.svg")
plt.show()

# -----------------------------------------------------------------------------
# Model Conversion
# -----------------------------------------------------------------------------
# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
model_no_quant_tflite = converter.convert()
# Save the model to disk
open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

# Convert the model to the TensorFlow Lite format with quantization
def representative_dataset():
  for _ in range(500):
    yield([feature_train.astype(np.float32)])
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
open(MODEL_TFLITE, "wb").write(model_tflite)

install_xxd()
convert_model_to_c_source(MODEL_TFLITE, MODEL_TFLITE_MICRO)
update_variable_names(MODEL_TFLITE_MICRO, MODEL_TFLITE)