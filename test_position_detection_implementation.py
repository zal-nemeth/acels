import serial
import time
import csv

# Adjust these variables as necessary
input_csv_path = 'acels/test_coordinates.csv'
output_csv_path = 'acels/output_coordinates.csv'
serial_port = 'COM4'
baud_rate = 9600

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)
time.sleep(2)  # Wait for the serial connection to initialize

with open(input_csv_path, mode='r') as infile, open(output_csv_path, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Skip header row in input, write header to output
    next(reader)
    writer.writerow(['x', 'y', 'z'])  # Assuming you only want x, y, z in the output

    for row in reader:
        # Send s1 through s8 as a comma-separated string, then read the response
        data_string = ','.join(row[:8]) + '\n'  # Only take first 8 columns
        ser.write(data_string.encode())

        # Read the Arduino's response
        response = ser.readline().decode().strip()
        if response:  # If there's a response, write it to the output CSV
            writer.writerow(response.split(','))

# Close the serial connection
ser.close()
