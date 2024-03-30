import os
import re
from collections import namedtuple

# Define a namedtuple for easier handling of model data
ModelData = namedtuple('ModelData', ['id', 'type', 'mae', 'runtime'])

def extract_data_from_filename(filename):
    """Extracts model ID and model type from the given filename."""
    match = re.search(r'(\d+)_model_(non_quant|quant)_impl_metrics\.txt', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def read_metrics(file_path):
    """Reads and extracts metrics from a given file."""
    with open(file_path, 'r') as file:
        contents = file.read()
        mae_match = re.search(r'# MAE: ([\d.]+) mm', contents)
        runtime_match = re.search(r'# Average Runtime: ([\d.]+) us', contents)
        if mae_match and runtime_match:
            return float(mae_match.group(1)), float(runtime_match.group(1))
    return None, None

def process_files(directory):
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
    with open(filename, 'w') as file:
        for model in sorted_models:
            file.write(f'Model ID: {model.id}, MAE: {model.mae}, Runtime: {model.runtime} us\n')

def main():
    directory = 'acels/metrics'
    models = process_files(directory)
    
    # Save top models based on criteria
    save_top_models(models, 'runtime', 'quant', 'acels/analysis/fastest_quant_models.txt')
    save_top_models(models, 'runtime', 'non_quant', 'acels/analysis/fastest_non_quant_models.txt')
    save_top_models(models, 'mae', 'quant', 'acels/analysis/most_accurate_quant_models.txt')
    save_top_models(models, 'mae', 'non_quant', 'acels/analysis/most_accurate_non_quant_models.txt')

if __name__ == "__main__":
    main()
