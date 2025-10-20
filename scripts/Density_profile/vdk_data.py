import os
import sys


# Define the grid dimensions
x_max = int(sys.argv[1])  # number of x slices
y_max = int(sys.argv[1])   # number of y slices
z_max = int(sys.argv[1])  # number of z slices

extracted_values = []

for k in range(z_max):          # z loop
    for j in range(y_max):      # y loop
        for i in range(x_max):  # x loop
            # Compute folder index
            folder_index = i + j * x_max + k * x_max * y_max
            folder_name = f"Eolder_{folder_index}"
            file_path = os.path.join(folder_name, "r.txt")

            try:
                with open(file_path, "r") as f:
                    # Extract the first number in the file and convert to float
                    value = float(f.read().strip())
                    extracted_values.append(value)
            except FileNotFoundError:
                print(f"{file_path} not found, appending 0")
                extracted_values.append(0)


#print("extracted_values:",extracted_values)

# Normalize the values by the maximum
max_value = max(extracted_values) if extracted_values else 1  # avoid division by zero
normalized_values = [val / max_value for val in extracted_values]

# Write the normalized list to a text file
output_file = "normalized_values.txt"
with open(output_file, "w") as f:
    for val in normalized_values:
        f.write(f"{val}\n")

print(f"Normalized values written to {output_file}")
