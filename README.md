# Attention Controlled Electromagnetic Levitation System


Welcome to the A.C.E.L.S package!

This package holds the tool that was developed to control an electromagnetic levitation system.


## Installation of Poetry
The installation instructions for poetry can be found [on their website](https://python-poetry.org/docs/).  Don't forget to add the new installation directory into your PATH.

## Installing the package using poetry

To install all dependencies for `acels` simply run `poetry install`. Some basic usage and commands for poetry can be found [here](https://python-poetry.org/docs/cli/) or by running `poetry help`.

With poetry, commands are executed inside the virtual environment and to signify this each command should be pre-pended with `poetry run`. For example to run pytest on the `acels` package you should run:
```sh
poetry run pytest
```
After `poetry run` you can specify any commands and arguments you need.

You can also run `poetry shell` which opens a poetry environment within the terminal. You can also integrate this into the Python interpreters in VS Code. Start by finding the path of the poetry's Python (for instance, by running `which python` on Linux). Then, input this path into the "Select Interpreter" section of VS Code settings.

## Folder Structure
```
├───acels/
│   ├───analysis/
│   ├───arduino/
│   │   ├───control/
│   │   └───position_detection/
│   ├───data/
│   ├───figures/
│   ├───metrics/
│   ├───models/
│   │   └───model/
│   │       └───variables/
│   ├───predictions/
├───docs/
└───tests/
```

## Features and Scripts

- **benchmarck_control_impl.py**: Drives entire control software with data input from CSV files, calculates average runtimes and metrics from predictions.
- **benchmark_matrix_trans_impl.py**: Benchmarks the matrix transformation algorithm with 3D coordinates input, providing average runtime performance.
- **benchmark_model_impl.py**: Evaluates position detection models (full, TFLite, and quantized TFLite) for accuracy and runtime, summarizing performance.
- **calc_mean_runtime.py**: Processes CSV files to calculate and average runtimes from specified columns.
- **create_impl_testing_coordinates.py**: Generates a dataset of 3D coordinates for testing matrix transformation algorithms.
- **data_analysis.py**: Analyzes metrics to identify top-performing models in terms of accuracy and speed, outputting results for both quantized and non-quantized models.
- **format_input_data.py**: Prepares data for Arduino processing, formatting it into the required array structure.
- **full_design_output_141.csv**: Contains metrics from a comprehensive implementation covering position detection and matrix transformation.
- **matrix_transformation.m**: MATLAB script for matrix transformation, used to generate C++ code.
- **matrix_transformation.py**: Python implementation of the matrix transformation algorithm for prototyping purposes.
- **plotting.py**: Provides functionalities for plotting metric data and tables.
- **position_detection_model.ipynb**: Jupyter notebook with a neural network prototype for position detection.
- **position_detection_nn.py**: Script for training position detection models, converting them for lite and C++ execution, and benchmarking.
- **precompute_matrix_transform_elements.py**: Optimizes matrix transformation by precomputing and hardcoding elements.
- **process_dataset.py**: Processes datasets to remove duplicates and clean data.

## Usage
### Arduino Integration

The `arduino` folder includes software for hardware integration for running and benchmarking the system. These scripts are designed for execution within the Arduino IDE and cover different aspects of the system:

- **acels_software_full**: Implements the full system integration, controlling the electromagnetic levitation process comprehensively including the position detection and the force-to-current matrix transfomration.
- **position_detection_non_quant**: Software for running and benchmarking non-quantized neural network models, focusing on position detection without model quantization.
- **position_detection_quantized**: Handles the quantized models, optimizing performance and accuracy in hardware implementation.

### Analysis
The `analysis` folder contains detailed data on runtime and accuracy metrics for various models. This includes information on the optimizers and activation functions used, along with lists of the top 10 fastest and most accurate models, both quantized and non-quantized.

### Metrics
Located in the `metrics` folder, you'll find performance metrics for both the original trained models and their integrated counterparts. This data is vital for understanding the effectiveness of the system and guiding further optimizations.

### Data
The `data` folder houses testing data and statistical analyses for each model. This information is used for evaluating model performance and for use in further training and validation processes.

### Figures
Training performance and final prediction plots are stored in the `figures` folder. These visuals are key for assessing training progress and comparing actual results against model predictions.

### Models
Trained models are kept within the `models` folder. This directory serves as a repository for all developed models, ensuring they are easily accessible for evaluation, benchmarking, and implementation.

### Predictions
Finally, the `predictions` folder stores the results from both PC-based runs and actual hardware implementation tests. This data provides insight into the practical effectiveness of the models in real-world scenarios.

### Note on Arduino Software
It is recommended to run the Arduino integration software directly from the Arduino IDE to ensure proper functionality and compatibility with the hardware.
