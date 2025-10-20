# Define input and output file names
input_file = 'martini_v3.0.0.itp'
output_file = 'Q1_C1_interactions.txt'

# Open input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Split the line into columns
        columns = line.strip().split()
        
        # Check if both 'Q1' and 'C1' are in the row (any order)
        if 'N4a' in columns and 'C1' in columns:
            outfile.write(line)
