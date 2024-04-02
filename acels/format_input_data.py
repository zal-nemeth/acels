import csv

# Replace 'yourfile.csv' with the path to your actual CSV file
csv_file_path = 'acels/data/xyz_impl_coordinates.csv'

with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    print("const float coordinateData[][3] = {")
    for row in csv_reader:
        # Assuming your CSV has columns named 'x', 'y', 'z'
        x, y, z = row['x'], row['y'], row['z']
        print(f"  {{{x}, {y}, {z}}},")
    print("};")
