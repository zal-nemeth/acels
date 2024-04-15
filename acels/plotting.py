# import pandas as pd
# import matplotlib.pyplot as plt

# # Example for one table
# df = pd.read_csv('acels/analysis/extended_50_mae_non_quant_models.csv')

# fig, ax = plt.subplots()
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# plt.savefig('table.svg', format='svg')


# import matplotlib.pyplot as plt
# import pandas as pd


# def create_svg_from_csv(csv_file, title):
#     # Read the CSV file
#     df = pd.read_csv(
#         csv_file, index_col=0
#     )  # Adjust if your index is differently located

#     # Creating the plot
#     fig, ax = plt.subplots()
#     ax.axis("tight")
#     ax.axis("off")
#     ax.table(
#         cellText=df.values.round(5),
#         colLabels=df.columns,
#         rowLabels=df.index,
#         cellLoc="center",
#         loc="center",
#     )

#     plt.title(title)

#     # Save the figure
#     svg_file = csv_file.replace(".csv", ".svg")
#     plt.savefig(svg_file, format="svg")
#     print(f"Saved SVG file: {svg_file}")


# # Example usage
# csv_file = "acels/analysis/extended_50_mae_non_quant_models.csv"  # Update this path
# title = "Your Title Here"  # Update this title
# create_svg_from_csv(csv_file, title)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
mae_non_quant = pd.read_csv("acels/analysis/extended_50_mae_non_quant_models.csv")
mae_quant = pd.read_csv("acels/analysis/extended_50_mae_quant_models.csv")
runtime_non_quant = pd.read_csv(
    "acels/analysis/extended_50_runtime_non_quant_models.csv"
)
runtime_quant = pd.read_csv("acels/analysis/extended_50_runtime_quant_models.csv")

import pandas as pd
import numpy as np


def load_and_prepare_data(csv_path, quantized=False):
    """
    Load the data from a CSV file and prepare it for plotting.

    Parameters:
    - csv_path: Path to the CSV file.
    - quantized: Boolean indicating whether the data is for quantized models.

    Returns:
    A dictionary where keys are optimizer names and values are lists of values for each activation function.
    """
    data = pd.read_csv(csv_path)
    data.fillna(
        0, inplace=True
    )  # Assuming missing values can be treated as 0s. Adjust if needed.
    results = {}
    for optimizer in ["Adam", "Adamax", "Nadam", "RMSprop"]:
        if quantized:
            key = f"Quant {optimizer}"
        else:
            key = f"Non-Quant {optimizer}"
        results[key] = data[optimizer].values.tolist()
    return results


def extract_data_for_plot(metric_type):
    """
    Extract data for plotting based on the metric type (MAE or Runtime).

    Parameters:
    - metric_type: 'MAE' or 'Runtime'

    Returns:
    Two lists of numpy arrays: one for non-quantized model data, another for quantized model data.
    """
    if metric_type == "MAE":
        non_quant_data = load_and_prepare_data(
            "acels/analysis/extended_50_mae_non_quant_models.csv"
        )
        quant_data = load_and_prepare_data(
            "acels/analysis/extended_50_mae_quant_models.csv", quantized=True
        )
    elif metric_type == "Runtime":
        non_quant_data = load_and_prepare_data(
            "acels/analysis/extended_50_runtime_non_quant_models.csv"
        )
        quant_data = load_and_prepare_data(
            "acels/analysis/extended_50_runtime_quant_models.csv", quantized=True
        )
    else:
        raise ValueError("Invalid metric_type. Choose either 'MAE' or 'Runtime'.")

    # Convert dictionary values to numpy arrays
    data_non_quant = [
        np.array(non_quant_data[f"Non-Quant {optimizer}"])
        for optimizer in ["Adam", "Adamax", "Nadam", "RMSprop"]
    ]
    data_quant = [
        np.array(quant_data[f"Quant {optimizer}"])
        for optimizer in ["Adam", "Adamax", "Nadam", "RMSprop"]
    ]

    return data_non_quant, data_quant


# Example plotting function using extract_data_for_plot
def plot_data(metric_type):
    data_non_quant, data_quant = extract_data_for_plot(metric_type)

    # Setup plot
    fig, axs = plt.subplots(figsize=(14, 7))
    categories = ["relu", "sigmoid", "swish", "tanh"]
    n_categories = len(categories)
    index = np.arange(n_categories)
    bar_width = 0.15

    # Plot configuration
    for i, optimizer in enumerate(["Adam", "Adamax", "Nadam", "RMSprop"]):
        axs.bar(
            index + i * bar_width,
            data_non_quant[i],
            bar_width,
            label=f"Non-Quant {optimizer}",
        )
        axs.bar(
            index + i * bar_width + bar_width * n_categories / 2,
            data_quant[i],
            bar_width,
            label=f"Quant {optimizer}",
            alpha=0.5,
        )

    axs.set_xlabel("Activation Function")
    if metric_type == "MAE":
        axs.set_ylabel("Mean Absolute Error (MAE)")
        axs.set_title("MAE by Activation Function and Optimizer Type")
    elif metric_type == "Runtime":
        axs.set_ylabel("Runtime (μs)")
        axs.set_title("Runtime by Activation Function and Optimizer Type")
    axs.set_xticks(index + bar_width * n_categories / 4)
    axs.set_xticklabels(categories)
    axs.legend()

    plt.tight_layout()
    plt.show()


# Plot MAE
# plot_data('MAE')
# plot_data('Runtime')

# def plot_combined(metric_types):
#     fig, axs = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
#     categories = ['relu', 'sigmoid', 'swish', 'tanh']
#     n_categories = len(categories)
#     index = np.arange(n_categories)  # base indices for categories
#     bar_width = 0.15
#     n_optimizers = 4  # Adjust based on the number of optimizers
#     total_width = n_optimizers * bar_width  # total width occupied by the grouped bars for an optimizer
#     offset = total_width / 2  # offset to center the groups

#     for metric_index, metric_type in enumerate(metric_types):
#         data_non_quant, data_quant = extract_data_for_plot(metric_type)

#         for i, optimizer in enumerate(['Adam', 'Adamax', 'Nadam', 'RMSprop']):
#             # Calculate offset for each optimizer group to be centered
#             pos_non_quant = index - offset + (i + 0.5) * bar_width
#             pos_quant = index - offset + (i + 0.5) * bar_width + bar_width / 2  # Slightly offset within the same group

#             axs[metric_index].bar(pos_non_quant, data_non_quant[i], bar_width * 0.9, label=f'Non-Quant {optimizer}' if metric_index == 0 else "", alpha=0.75)
#             axs[metric_index].bar(pos_quant, data_quant[i], bar_width * 0.9, label=f'Quant {optimizer}' if metric_index == 0 else "", alpha=0.5, linestyle='--')

#         axs[metric_index].set_ylabel(metric_type)
#         axs[metric_index].set_title(f'{metric_type} by Activation Function and Optimizer Type')
#         axs[metric_index].set_xticks(index)
#         axs[metric_index].set_xticklabels(categories)

#     axs[1].set_xlabel('Activation Function')
#     axs[0].legend(loc='upper left', bbox_to_anchor=(1,1))  # Adjust legend location if needed

#     plt.tight_layout()
#     plt.show()


def plot_combined_v0(metric_types):
    # Setup figure for subplots: 2 rows for MAE and Runtime, 1 column
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    categories = ["relu", "sigmoid", "swish", "tanh"]
    n_categories = len(categories)
    index = np.arange(n_categories)
    bar_width = 0.15
    colors = {}

    for metric_index, metric_type in enumerate(metric_types):
        data_non_quant, data_quant = extract_data_for_plot(metric_type)

        for i, optimizer in enumerate(["Adam", "Adamax", "Nadam", "RMSprop"]):
            non_quant_bar = axs[metric_index].bar(
                index + i * bar_width,
                data_non_quant[i],
                bar_width,
                label=f"Non-Quant {optimizer}" if metric_index == 0 else "",
                alpha=0.75,
            )
            quant_bar = axs[metric_index].bar(
                index + i * bar_width + bar_width * n_categories / 2,
                data_quant[i],
                bar_width,
                label=f"Quant {optimizer}" if metric_index == 0 else "",
                alpha=0.5,
            )
            colors[f"Non-Quant {optimizer}"] = mcolors.to_hex(non_quant_bar[0].get_facecolor())
            colors[f"Quant {optimizer}"] = mcolors.to_hex(quant_bar[0].get_facecolor())


        axs[metric_index].set_ylabel(metric_type)
        axs[metric_index].set_title(
            f"{metric_type} by Activation Function and Optimizer Type"
        )
        axs[metric_index].set_xticks(index + bar_width * n_categories / 3)
        axs[metric_index].set_xticklabels(categories)

    axs[1].set_xlabel("Activation Function")
    axs[0].legend(loc="best")  # Show legend only on the first plot

    plt.tight_layout()
    plt.show()

    return colors

def plot_combined(metric_types):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    categories = ["relu", "sigmoid", "swish", "tanh"]
    n_categories = len(categories)
    index = np.arange(n_categories)
    bar_width = 0.2

    # Predefined colors for each bar type
    colors = {'Non-Quant Adam': '#8fbbda', 'Quant Adam': '#ffbf86', 'Non-Quant Adamax': '#96d096', 'Quant Adamax': '#ea9394', 'Non-Quant Nadam': '#cab3de', 'Quant Nadam': '#c6aaa5', 'Non-Quant RMSprop': '#f1bbe0', 'Quant RMSprop': '#bfbfbf'}
    colors = {'Non-Quant Adam': '#79add2', 'Quant Adam': '#ffb26e', 'Non-Quant Adamax': '#80c680', 'Quant Adamax': '#e67d7e', 'Non-Quant Nadam': '#bfa4d7', 'Quant Nadam': '#ba9a93', 'Non-Quant RMSprop': '#eeadda', 'Quant RMSprop': '#b2b2b2'}
    for metric_index, metric_type in enumerate(metric_types):
        data_non_quant, data_quant = extract_data_for_plot(metric_type)
        plotted_labels = set()

        for i, optimizer in enumerate(["Adam", "Adamax", "Nadam", "RMSprop"]):
            for j, category in enumerate(categories):
                non_quant_value = data_non_quant[i][j]
                quant_value = data_quant[i][j]

                # Determine which bar should be in front based on the smaller value
                front = "Quant" if quant_value < non_quant_value else "Non-Quant"

                # Define labels for legend control
                non_quant_label = f"{optimizer} (non-quant)"
                quant_label = f"{optimizer} (quant)"

                # Plot the bar that should be in the back first
                if front == "Quant":
                    axs[metric_index].bar(
                        index[j] + i * bar_width,
                        non_quant_value,
                        bar_width,
                        label=non_quant_label if non_quant_label not in plotted_labels else "",
                        color=colors[f'Non-Quant {optimizer}'],
                        alpha=1
                    )
                    axs[metric_index].bar(
                        index[j] + i * bar_width,
                        quant_value,
                        bar_width,
                        label=quant_label if quant_label not in plotted_labels else "",
                        color=colors[f'Quant {optimizer}'],
                        alpha=1
                    )
                else:
                    axs[metric_index].bar(
                        index[j] + i * bar_width,
                        quant_value,
                        bar_width,
                        label=quant_label if quant_label not in plotted_labels else "",
                        color=colors[f'Quant {optimizer}'],
                        alpha=1
                    )
                    axs[metric_index].bar(
                        index[j] + i * bar_width,
                        non_quant_value,
                        bar_width,
                        label=non_quant_label if non_quant_label not in plotted_labels else "",
                        color=colors[f'Non-Quant {optimizer}'],
                        alpha=1
                    )

                # Mark these labels as plotted to avoid duplicate legend entries
                plotted_labels.update([non_quant_label, quant_label])

        if metric_type == "MAE":
            axs[metric_index].set_ylabel(f"{metric_type} (mm)")
        else:
            axs[metric_index].set_ylabel(f"{metric_type} (μs)")
            
        axs[metric_index].set_title(f"{metric_type} by Activation Function and Optimizer Type")
        axs[metric_index].set_xticks(index + bar_width * n_categories / 2.65)
        axs[metric_index].set_xticklabels(categories)

    axs[1].set_xlabel("Activation Function")
    axs[0].legend(loc="best")  # Show legend only on the first plot, no duplicates

    plt.tight_layout()
    plt.show()

    return colors

# Plot both MAE and Runtime on the same figure
# colours_used = plot_combined(["MAE", "Runtime"])
# print(colours_used)


import pandas as pd
from tabulate import tabulate

def load_and_display_data(file_path):
    # Load data from a CSV file into a DataFrame
    data = pd.read_csv(file_path, index_col='Activation')
    # Display the DataFrame using tabulate for a table-like format
    print(tabulate(data, headers='keys', tablefmt='psql'))
    print("\n")  # Add a newline for better separation of tables

# Paths to the CSV files
files = [
    'acels/analysis/extended_50_mae_non_quant_models.csv',
    'acels/analysis/extended_50_mae_quant_models.csv',
    'acels/analysis/extended_50_runtime_non_quant_models.csv',
    'acels/analysis/extended_50_runtime_quant_models.csv',
    'acels/analysis/extended_150_mae_non_quant_models.csv',
    'acels/analysis/extended_150_mae_quant_models.csv',
    'acels/analysis/extended_150_runtime_non_quant_models.csv',
    'acels/analysis/extended_150_runtime_quant_models.csv',
    'acels/analysis/trimmed_50_mae_non_quant_models.csv',
    'acels/analysis/trimmed_50_mae_quant_models.csv',
    'acels/analysis/trimmed_50_runtime_non_quant_models.csv',
    'acels/analysis/trimmed_50_runtime_quant_models.csv',
    'acels/analysis/trimmed_150_mae_non_quant_models.csv',
    'acels/analysis/trimmed_150_mae_quant_models.csv',
    'acels/analysis/trimmed_150_runtime_non_quant_models.csv',
    'acels/analysis/trimmed_150_runtime_quant_models.csv'
]


# def save_data_as_svg(dataframe, filename):
#     # Create a plot object with dataframe data
#     fig, ax = plt.subplots()
#     # Remove the x and y axis
#     ax.axis('off')
#     ax.axis('tight')
#     # Prepare data and headers for the table
#     cell_text = dataframe.values
#     col_labels = ['Activation'] + dataframe.columns.tolist()
#     row_labels = dataframe.index.tolist()
    
#     # Create an array including the row labels as the first column of cell_text
#     full_cell_text = [[row_label] + list(row) for row_label, row in zip(row_labels, cell_text)]
    
#     # Table added to plot with adjusted row labels and headers
#     table = ax.table(
#         cellText=full_cell_text,
#         colLabels=col_labels,
#         loc='center',
#         cellLoc='center'
#     )
    
#     # Adjust layout to make room for table
#     fig.tight_layout()
#     # Save the table as an SVG file
#     plt.savefig(f"{filename}.svg", format='svg')

# def save_data_as_excel(dataframe, filename):
#     # Save the dataframe to an Excel file
#     dataframe.to_excel(f"{filename}.xlsx")

# def process_and_save(file_path):
#     # Load data from CSV
#     data = pd.read_csv(file_path, index_col='Activation')
#     # Save as SVG and Excel
#     base_filename = file_path.split('.')[0]  # Remove extension from file name
#     save_data_as_svg(data, base_filename)
#     save_data_as_excel(data, base_filename)


# # Process each file
# for file in files:
#     process_and_save(file)

import pandas as pd
import matplotlib.pyplot as plt

def add_units(dataframe, data_type):
    # Define unit based on the data type
    unit = "mm" if "mae" in data_type.lower() else "us"
    # Convert data to string and append unit
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].apply(lambda x: f"{x:.6f} {unit}" if pd.notna(x) else "")
    return dataframe

def save_data_as_svg(dataframe, filename):
    # Create a plot object with dataframe data
    fig, ax = plt.subplots()
    # Remove the x and y axis
    ax.axis('off')
    ax.axis('tight')
    # Prepare data and headers for the table
    cell_text = dataframe.values
    col_labels = ['Activation'] + dataframe.columns.tolist()
    row_labels = dataframe.index.tolist()
    
    # Create an array including the row labels as the first column of cell_text
    full_cell_text = [[row_label] + list(row) for row_label, row in zip(row_labels, cell_text)]
    
    # Table added to plot with adjusted row labels and headers
    table = ax.table(
        cellText=full_cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Adjust layout to make room for table
    fig.tight_layout()
    # Save the table as an SVG file
    plt.savefig(f"{filename}.svg", format='svg')

def save_data_as_excel(dataframe, filename):
    # Save the dataframe to an Excel file
    dataframe.to_excel(f"{filename}.xlsx")

def process_and_save(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path, index_col='Activation')
    # Determine if the data is for MAE or runtime based on the filename
    data_type = "mae" if "mae" in file_path.lower() else "runtime"
    # Add units to the dataframe
    data_with_units = add_units(data, data_type)
    # Save as SVG and Excel
    base_filename = file_path.split('.')[0]  # Remove extension from file name
    save_data_as_svg(data_with_units, base_filename)
    save_data_as_excel(data_with_units, base_filename)

# Process each file
for file in files:
    process_and_save(file)


# Display each table
# for file in files:
#     print(f"Displaying data for: {file}")
#     load_and_display_data(file)
    
# def save_data_as_svg(dataframe, filename):
#     # Create a plot object with dataframe data
#     fig, ax = plt.subplots()
#     # Remove the x and y axis
#     ax.axis('off')
#     ax.axis('tight')
#     # Table added to plot
#     ax.table(cellText=dataframe.values, colLabels=dataframe.columns, rowLabels=dataframe.index, loc='center', cellLoc='center')
#     # Adjust layout to make room for table
#     fig.tight_layout()
#     # Save the table as an SVG file
#     plt.savefig(f"{filename}.svg", format='svg')

# def save_data_as_excel(dataframe, filename):
#     # Save the dataframe to an Excel file
#     dataframe.to_excel(f"{filename}.xlsx")

# def process_and_save(file_path):
#     # Load data from CSV
#     data = pd.read_csv(file_path, index_col='Activation')
#     # Save as SVG and Excel
#     base_filename = file_path.split('.')[0]  # Remove extension from file name
#     save_data_as_svg(data, base_filename)
#     save_data_as_excel(data, base_filename)
