import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Load the universe with the topology and trajectory files
u = mda.Universe("cyl.tpr", "traj_comp.xtc")

# Select all atoms (assuming PO4 are the atoms of interest)
ag = u.select_atoms('name C3B')  # Positions are in Angstroms

cylinder_x_min = np.min(ag.positions[:, 0])
cylinder_x_max = np.max(ag.positions[:, 0])
print("cylinder length (A):",cylinder_x_max)
segment_width = 20  # Width of each segment in Angstroms

# Calculate the number of segments
num_segments = int((cylinder_x_max - cylinder_x_min) / segment_width)


mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments)}
std_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments)}
max_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments)}

# Iterate over trajectory frames
for ts in u.trajectory:
    print(f"Processing Frame {ts.frame}")
    for segment in range(num_segments):
        #print("segment:\n",segment)
        segment_x_min = cylinder_x_min + segment * segment_width
        #print("segment_x_min:\n",segment_x_min)
        segment_x_max = segment_x_min + segment_width
        #print("segment_x_max:\n",segment_x_max)
        segment_ag = ag.select_atoms(f"prop x >= {segment_x_min} and prop x < {segment_x_max}")
        #print((segment_ag.positions))
        #print((segment_ag.positions[:,0]))
        center_of_mass = segment_ag.center_of_mass()
        #print(center_of_mass)

        
        #distances  = np.linalg.norm(segment_ag.positions - center_of_mass, axis=1)
        distances = np.linalg.norm(segment_ag.positions[:, 1:3] - center_of_mass[1:3], axis=1) # ignoring x coordinate

        
        
        
        
        #print((distances))
        MEAN_SEGMENT = np.mean(distances)
        #print("MEAN_SEGMENT:",MEAN_SEGMENT)
        MAX_SEGMENT = np.max(distances)
        #print("MAX_SEGMENT:",MAX_SEGMENT)
        MIN_SEGMENT = np.min(distances)
        #print("MIN_SEGMENT:",MIN_SEGMENT)
        mean_segment_values[f'Segment_{segment+1}'].append(MEAN_SEGMENT)
        max_segment_values[f'Segment_{segment+1}'].append(MAX_SEGMENT)
        #print("\n\n\n")
        #print("\n\n\n")


# Create a DataFrame from the mean segment values
mean_segment_df = pd.DataFrame(mean_segment_values)
print(mean_segment_df)
mean_segment_df.to_csv('mean_segment_df_V2.txt', sep='\t', index=False)










######################################################################################################



# column_means = mean_segment_df.mean(axis=0)
# mean_segment_df.loc["mean"] = column_means
# # Extract the last row (means)
# last_row = mean_segment_df.loc["mean"]
# 
# # Write to file with format: column_index (1-based)   mean_value
# with open("column_means_row.txt", "w") as f:
#     for i, val in enumerate(last_row, start=1):
#         f.write(f"{i}  {val:.6f}\n")
#         
# # Create the DataFrame
# mean_segment_df = pd.DataFrame(mean_segment_values)
# 
# # Compute and append column-wise means
# column_means = mean_segment_df.mean(axis=0)
# mean_segment_df.loc["mean"] = column_means
# 
# # Extract the last row (the means)
# last_row = mean_segment_df.loc["mean"]
# 
# # Prepare data for plotting
# x = list(range(1, len(last_row) + 1))  
# y = last_row.values
# 
# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, marker='o', linestyle='-')
# plt.ylim(50, 60)  # Set y-axis limits
# plt.xlabel("Segment Index")
# plt.ylabel("Mean Radius")
# plt.savefig("RADIUS_MEAN.png", dpi=300)
 


######################################################################################################
# mean_segment_df = pd.DataFrame(mean_segment_values)
# mean_segment_df["mean"] = mean_segment_df.mean(axis=1)
# # Create a new DataFrame with row numbers starting from 1
# mean_only_df = pd.DataFrame({
#     "row": range(1, len(mean_segment_df) + 1),
#     "mean": mean_segment_df["mean"]
# })
# 
# # Write to a text file
# mean_only_df.to_csv("mean_only.txt", sep="\t", index=False, header=False, float_format="%.3f")
# 
# plt.figure(figsize=(8, 5))
# plt.plot(mean_only_df["row"], mean_only_df["mean"], marker='o', linestyle='-')
# plt.xlabel("Frame Number")
# plt.ylabel("Mean Radius")
# plt.ylim(50, 60)  # Set y-axis limits
# # Save the figure
# plt.tight_layout()
# plt.savefig("RADIUS_MEAN.png", dpi=300)
######################################################################################################

print("\n\n\n")
# Number of segments
num_segments = len(mean_segment_values)
fig, axes = plt.subplots(nrows=int(np.ceil(num_segments / 3)), ncols=3, figsize=(15, 10))
axes = axes.flatten()
segment_means = []  # <-- list to store mean values

for i, segment in enumerate(mean_segment_values.keys()):
    data = mean_segment_df[segment].dropna()
    mean = data.mean()
    std_dev = data.std()
    segment_means.append(mean)  # <-- save mean
    axes[i].hist(data, bins=20, density=True, alpha=0.6, color='gray', label='Histogram')
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
    y = norm.pdf(x, mean, std_dev)
    axes[i].plot(x, y, color='blue')
    axes[i].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    axes[i].axvline(mean + std_dev, color='green', linestyle='--', label=f'St. Dev.: {std_dev:.2f}')
    axes[i].axvline(mean - std_dev, color='green', linestyle='--')
    axes[i].set_title(f'{segment}')
    axes[i].set_xlabel('R')
    axes[i].set_ylabel('PD')
    axes[i].legend(loc='upper right')

# Turn off unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.savefig('Radius_Mean_Histogram.png', dpi=300)
plt.close()

# --- Plot mean vs. segment index ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_segments + 1), segment_means, marker='o', linestyle='-')
plt.xlabel("Segment Index")
plt.ylabel("Mean Radius")
plt.ylim(40, 70)  # Set y-axis limits
plt.title("Mean Radius per Segment")
plt.savefig("Radius_Mean_VS_SegNumber.png", dpi=300)




print(" \n The guassian distribution of radius for each segment is generated for the entire trajectory!\n" )
print(" The mean value of the rafius for each frame is calculated as follows:\n" )
print(" 1. Center of mass of all C3B particles is calculated.\n")
print(" 2. Distance of each C3B particle is caculated with respect to the COM excluding x dimention.\n")
print(" 3. Mean of the distance values is represented as the radius of the system.\n")

plt.tight_layout()





























