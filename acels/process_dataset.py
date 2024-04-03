import pandas as pd

# Load your CSV file
df = pd.read_csv("acels/data/position_data_float_xyz_extended.csv")

# Remove duplicate rows
df_unique = df.drop_duplicates()

# Save to a new CSV file
df_unique.to_csv("acels/data/position_dataset_trimmed.csv", index=False)
