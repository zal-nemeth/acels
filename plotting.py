import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data = {
    ("Non-Quantized", "MAE", "Adam"): [0.06910252, 0.0559788, 0.05089507, 0.04853391],
    ("Non-Quantized", "MAE", "Adamax"): [0.07557162, None, 0.07885556, 0.05022648],
    ("Non-Quantized", "MAE", "Nadam"): [0.05243826, 0.03764004, 0.07225274, 0.04992772],
    ("Non-Quantized", "MAE", "RMSprop"): [None, None, None, 0.0739429],
    ("Quantized", "MAE", "Adam"): [0.13639561, 0.18565595, 0.19484881, 0.07732683],
    ("Quantized", "MAE", "Adamax"): [0.22800747, None, 0.30997832, 0.14761595],
    ("Quantized", "MAE", "Nadam"): [0.13551259, 0.16124684, 0.25240332, 0.09844718],
    ("Quantized", "MAE", "RMSprop"): [None, None, None, 0.11371281],
    ("Non-Quantized", "Runtime", "Adam"): [
        515.04228406,
        631.26129382,
        678.48680882,
        694.27755692,
    ],
    ("Non-Quantized", "Runtime", "Adamax"): [
        514.21864836,
        None,
        678.64329599,
        690.67112396,
    ],
    ("Non-Quantized", "Runtime", "Nadam"): [
        514.23021323,
        633.04228406,
        678.55908927,
        692.55655945,
    ],
    ("Non-Quantized", "Runtime", "RMSprop"): [None, None, None, 693.096133],
    ("Quantized", "Runtime", "Adam"): [
        164.15034333,
        759.35634261,
        1040.7426816,
        1130.9194073,
    ],
    ("Quantized", "Runtime", "Adamax"): [
        163.54282617,
        None,
        1095.14745211,
        1246.31189013,
    ],
    ("Quantized", "Runtime", "Nadam"): [
        163.83556198,
        848.56812432,
        1068.72135887,
        1195.25984821,
    ],
    ("Quantized", "Runtime", "RMSprop"): [None, None, None, 1147.3230936],
}

# Create a multi-index for columns
columns = pd.MultiIndex.from_tuples(
    data.keys(), names=["Model Type", "Metric", "Optimizer"]
)

# Create DataFrame
df = pd.DataFrame(data, columns=columns, index=["relu", "sigmoid", "swish", "tanh"])

print(df)

# Prepare data for MAE and Runtime for both Non-Quantized and Quantized models
categories = df.index.tolist()  # Activation functions
n_categories = len(categories)
bar_width = 0.2
index = np.arange(n_categories)

# Function to extract data for plotting
def extract_data_for_plot(metric):
    data_non_quant = df["Non-Quantized", metric].values
    data_quant = df["Quantized", metric].values
    return data_non_quant.T, data_quant.T


# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 10), tight_layout=True)

# MAE plot
data_non_quant_mae, data_quant_mae = extract_data_for_plot("MAE")
for i, optimizer in enumerate(["Adam", "Adamax", "Nadam", "RMSprop"]):
    axs[0].bar(
        index + i * bar_width,
        data_non_quant_mae[i],
        bar_width,
        label=f"Non-Quant {optimizer}",
    )
    axs[0].bar(
        index + i * bar_width + bar_width * n_categories / 2,
        data_quant_mae[i],
        bar_width,
        label=f"Quant {optimizer}",
    )

axs[0].set_xlabel("Activation Function")
axs[0].set_ylabel("Mean Absolute Error (MAE)")
axs[0].set_title("MAE by Activation Function and Optimizer Type")
axs[0].set_xticks(index + bar_width / 2 * (n_categories - 1))
axs[0].set_xticklabels(categories)
axs[0].legend()

# Runtime plot
data_non_quant_runtime, data_quant_runtime = extract_data_for_plot("Runtime")
for i, optimizer in enumerate(["Adam", "Adamax", "Nadam", "RMSprop"]):
    axs[1].bar(
        index + i * bar_width,
        data_non_quant_runtime[i],
        bar_width,
        label=f"Non-Quant {optimizer}",
    )
    axs[1].bar(
        index + i * bar_width + bar_width * n_categories / 2,
        data_quant_runtime[i],
        bar_width,
        label=f"Quant {optimizer}",
    )

axs[1].set_xlabel("Activation Function")
axs[1].set_ylabel("Runtime (seconds)")
axs[1].set_title("Runtime by Activation Function and Optimizer Type")
axs[1].set_xticks(index + bar_width / 2 * (n_categories - 1))
axs[1].set_xticklabels(categories)
axs[1].legend()

plt.show()
