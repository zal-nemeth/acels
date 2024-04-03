import csv


def calculate_average_runtime(filename):
    total_runtime = 0
    count = 0

    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Assuming runtime is numeric and directly convertible
                total_runtime += float(row["runtime"])
                count += 1
            except ValueError:
                # Handles cases where runtime is missing or not a number
                continue

    if count > 0:
        average_runtime = total_runtime / count
        print(f"Average Runtime: {average_runtime}")
    else:
        print("No valid runtime data found.")


# Example usage
# Replace 'your_file.csv' with the path to your actual CSV file
calculate_average_runtime("acels/output_runtime.csv")
