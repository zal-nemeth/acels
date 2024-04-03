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

To use the A.C.E.L.S package, please refer to the individual script documentation for detailed instructions on each tool's purpose and usage.
