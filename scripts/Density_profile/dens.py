import os
import numpy as np
import sys


NUM_files =  int(sys.argv[1])
values = []

for i in range(NUM_files):
    gro_file = f"grid_{i}.gro"
    empty_file = f"EMPTY_{i}.txt"
    
    if os.path.exists(gro_file):
        try:
            with open(gro_file, "r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    value = int(lines[1].strip())
                    values.append(value)
                else:
                    #print(f"{gro_file} does not have a second line, appending 0")
                    values.append(0)
        except Exception as e:
            print(f"Error reading {gro_file}: {e}, appending 0")
            values.append(0)
    elif os.path.exists(empty_file):
        values.append(0)
    else:
        print(f"Neither {gro_file} nor {empty_file} found, appending 0")
        values.append(0)

# Compute mean
#print(values)
mean_value = int(np.mean(values))

# Write mean to r.txt
with open("r.txt", "w") as f:
    f.write(f"{mean_value}\n")

#print(f"Mean value {mean_value} written to r.txt")
