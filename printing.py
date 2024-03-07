import pandas as pd

# Load the CSV
df = pd.read_csv('/home/zalnemeth/repos/acels/acels/data/r_z_data_1A.csv')

# Select the column (replace 'column_name' with your actual column name)
column_data1 = df['Fx']
column_data2 = df['Fy']
column_data3 = df['Fz']
column_data4 = df['Tx']
column_data5 = df['Tx']
column_data6 = df['r']
column_data7 = df['z']

# Convert to string separated by commas
row_string = ','.join(column_data1.astype(str))

# Print or save to file
print(len(column_data1))
print(len(column_data2))
print(len(column_data3))
print(len(column_data4))
print(len(column_data5))
print(len(column_data6))
print(len(column_data7))
