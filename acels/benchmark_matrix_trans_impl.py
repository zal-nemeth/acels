import csv
import struct
import time

import serial


def send_xyz_and_get_average_runtime(input_csv_path):
    ser = serial.Serial("COM4", 9600)
    time.sleep(2)  # Wait for the Arduino to reset

    # Function to send x, y, z values to Arduino
    def send_values_to_arduino(x, y, z):
        ser.write(struct.pack("ddd", x, y, z))
        time.sleep(0.1)  # A short delay to ensure Arduino processes data

    # Read x, y, z from CSV file and send them to Arduino
    with open(input_csv_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)  # Skip the header row
        for row in csvreader:
            x, y, z = map(float, row[-3:])  # Get the last three columns as floats
            send_values_to_arduino(x, y, z)

    time.sleep(2)  # Wait for Arduino to process and send back the data

    # Check if there is data waiting to be read
    if ser.inWaiting() >= 8:  # Expecting a double value
        average_time = struct.unpack("d", ser.read(8))[0]
        print(f"Average Execution Time: {average_time} microseconds")

        # Save the average execution time to a text file
        with open("average_runtime.txt", "w") as f:
            f.write(f"{average_time}\n")

    ser.close()


if __name__ == "__main__":
    send_xyz_and_get_average_runtime("acels/data/xyz_impl_coordinates.csv")
