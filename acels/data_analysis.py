import os
import re
from collections import namedtuple

import pandas as pd

# Define a namedtuple for easier handling of model data
ModelData = namedtuple("ModelData", ["id", "type", "mae", "runtime"])


def extract_data_from_filename(filename):
    """Extracts model ID and model type from the given filename."""
    match = re.search(r"(\d+)_model_(non_quant|quant)_impl_metrics\.txt", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def read_metrics(file_path):
    """Reads and extracts metrics from a given file."""
    with open(file_path, "r") as file:
        contents = file.read()
        mae_match = re.search(r"# MAE: ([\d.]+) mm", contents)
        runtime_match = re.search(r"# Average Runtime: ([\d.]+) us", contents)
        if mae_match and runtime_match:
            return float(mae_match.group(1)), float(runtime_match.group(1))
    return None, None


def process_files_top(directory):
    """Processes all files in the directory and organizes data."""
    models = []
    for filename in os.listdir(directory):
        if filename.endswith("_impl_metrics.txt"):
            model_id, model_type = extract_data_from_filename(filename)
            if model_id and model_type:
                mae, runtime = read_metrics(os.path.join(directory, filename))
                if mae is not None and runtime is not None:
                    models.append(ModelData(model_id, model_type, mae, runtime))
    return models


def save_top_models(models, criterion, model_type, filename):
    """Saves top 10 models to a file based on the given criterion and model type."""
    filtered_models = [m for m in models if m.type == model_type]
    sorted_models = sorted(filtered_models, key=lambda x: getattr(x, criterion))[:10]
    with open(filename, "w") as file:
        for model in sorted_models:
            file.write(
                f"Model ID: {model.id}, MAE: {model.mae}, Runtime: {model.runtime} us\n"
            )


# Structure to hold model data
class ModelInfo:
    def __init__(self, model_id, activation, optimizer, patience, dataset_type):
        self.model_id = model_id
        self.activation = activation
        self.optimizer = optimizer
        self.patience = patience
        self.dataset_type = dataset_type
        # Metrics from the implementation metrics file
        self.mae_quant = None
        self.runtime_quant = None
        self.mae_non_quant = None
        self.runtime_non_quant = None


def extract_og_metrics(filename):
    """Extracts needed information from the original metrics file."""
    with open(filename, "r") as file:
        content = file.read()

        # Use a default value if the pattern is not found
        activation_search = re.search(r"Layer: \w+, Activation: (\w+)", content)
        activation = activation_search.group(1) if activation_search else "unknown"

        optimizer_search = re.search(r"Optimizer: (\w+)", content)
        optimizer = optimizer_search.group(1) if optimizer_search else "unknown"

        patience_search = re.search(r"Patience: (\d+)", content)
        patience = (
            int(patience_search.group(1)) if patience_search else 0
        )  # Using 0 or another default value

        dataset_type_search = re.search(r"Dataset: (\w+)", content)
        dataset_type = (
            dataset_type_search.group(1) if dataset_type_search else "unknown"
        )

    return activation, optimizer, patience, dataset_type


def read_impl_metrics(filename):
    """Reads implementation metrics to extract MAE and runtime."""
    with open(filename, "r") as file:
        contents = file.read()
        mae = float(re.search(r"# MAE: ([\d.]+)", contents).group(1))
        runtime = float(re.search(r"# Average Runtime: ([\d.]+)", contents).group(1))
    return mae, runtime


def process_files(directory):
    models = {}
    # Process original metrics files
    for filename in os.listdir(directory):
        if filename.endswith("_og_metrics.txt"):
            model_id = filename.split("_")[0]
            # print(model_id)
            filepath = os.path.join(directory, filename)
            activation, optimizer, patience, dataset_type = extract_og_metrics(filepath)
            models[model_id] = ModelInfo(
                model_id, activation, optimizer, patience, dataset_type
            )
            if model_id == "339":
                print(patience)
            

    # Process implementation metrics files
    for filename in os.listdir(directory):
        if "_impl_metrics.txt" in filename:
            model_id = filename.split("_")[0]
            # Correct approach: Check for 'non_quant' first
            if "_non_quant_" in filename:
                model_type = "non_quant"
            elif "_quant_" in filename:  # If it's not non_quant, then it must be quant
                model_type = "quant"
            else:
                continue  # Skip if the filename doesn't match expected patterns

            filepath = os.path.join(directory, filename)
            mae, runtime = read_impl_metrics(filepath)
            if model_type == "quant":
                models[model_id].mae_quant = mae
                models[model_id].runtime_quant = runtime
            else:  # This now correctly handles non_quant models
                models[model_id].mae_non_quant = mae
                models[model_id].runtime_non_quant = runtime

    return models


def create_and_save_tables(models):
    # Convert models dictionary to DataFrame
    data = {
        "Model ID": [],
        "Activation": [],
        "Optimizer": [],
        "Patience": [],
        "Dataset Type": [],
        "MAE Quant": [],
        "Runtime Quant": [],
        "MAE Non Quant": [],
        "Runtime Non Quant": [],
    }

    for model_id, info in models.items():
        data["Model ID"].append(model_id)
        data["Activation"].append(info.activation)
        data["Optimizer"].append(info.optimizer)
        data["Patience"].append(info.patience)
        data["Dataset Type"].append(info.dataset_type)
        data["MAE Quant"].append(info.mae_quant)
        data["Runtime Quant"].append(info.runtime_quant)
        data["MAE Non Quant"].append(info.mae_non_quant)
        data["Runtime Non Quant"].append(info.runtime_non_quant)

    df = pd.DataFrame(data)

    # Ensure correct handling for both quant and non-quant models
    for dataset_type in ["extended", "trimmed"]:
        for patience in [50, 150, 200, 250, 500]:
            for metric in ["MAE", "Runtime"]:
                # Ensure we handle both quant and non-quant correctly by specifying the correct column names
                for model_type in ["Quant", "Non Quant"]:
                    if model_type == "Quant":
                        metric_column = f"{metric} Quant"
                    else:
                        metric_column = f"{metric} Non Quant"

                    filtered_df = df[
                        (df["Dataset Type"] == dataset_type)
                        & (df["Patience"] == patience)
                    ]

                    # Make sure there's data to pivot
                    if not filtered_df.empty:
                        table = pd.pivot_table(
                            filtered_df,
                            values=metric_column,
                            index=["Activation"],
                            columns=["Optimizer"],
                            aggfunc="mean",  # Use 'mean', 'first', or another appropriate aggregation function
                        )

                        # Save to CSV only if the pivot table isn't empty
                        if not table.empty:
                            filename = f'acels/analysis/{dataset_type}_{patience}_{metric.lower()}_{model_type.lower().replace(" ", "_")}_models.csv'
                            table.to_csv(filename)
                            print(f"Saved: {filename}")
                        else:
                            print(
                                f'No data for table: {dataset_type}_{patience}_{metric.lower()}_{model_type.lower().replace(" ", "_")}'
                            )
                    else:
                        print(
                            f"Filtered DataFrame is empty for: {dataset_type}, {patience}, {metric}, {model_type}"
                        )


if __name__ == "__main__":
    directory = "acels/metrics"
    top_models = process_files_top(directory)

    # Save top models based on criteria
    save_top_models(
        top_models, "runtime", "quant", "acels/analysis/fastest_quant_models.txt"
    )
    save_top_models(
        top_models,
        "runtime",
        "non_quant",
        "acels/analysis/fastest_non_quant_models.txt",
    )
    save_top_models(
        top_models, "mae", "quant", "acels/analysis/most_accurate_quant_models.txt"
    )
    save_top_models(
        top_models,
        "mae",
        "non_quant",
        "acels/analysis/most_accurate_non_quant_models.txt",
    )

    models = process_files(directory)
    create_and_save_tables(models)
