import pandas as pd

# Define ranges for x, y, and z
x_range = range(-5, 6)
y_range = range(-5, 6)
z_range = range(2, 9)

# Create a list of dictionaries, each representing a point in 3D space
points = [{"x": x, "y": y, "z": z} for x in x_range for y in y_range for z in z_range]

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(points)

# Specify the file path
file_path = "acels/data/xyz_impl_coordinates.csv"

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

file_path
