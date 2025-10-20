import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load system
u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")

segment_width = 20  # Angstroms

# First pass: Determine global min/max z
z_min_global = np.inf
z_max_global = -np.inf


FRAME_ANALYSIS=10000

for ts in u.trajectory:
    ag = u.select_atoms('name C3B')
    z_min_global = min(z_min_global, np.min(ag.positions[:, 2]))
    z_max_global = max(z_max_global, np.max(ag.positions[:, 2]))

cylinder_length_global = z_max_global - z_min_global
num_segments_global = int(np.ceil(cylinder_length_global / segment_width)) # np.ceil() rounds that number up to the nearest integer
print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
print(f"Fixed number of segments: {num_segments_global}")

# Initialize storage for segment stats
mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}

# Store frame index + constriction radius for later binning
frame_radius_pairs = []

# Iterate over frames
for ts in u.trajectory:
    print(f"Processing Frame {ts.frame}")
    ag = u.select_atoms('name C3B')
    
    box = ts.dimensions 
    Lx, Ly, Lz = box[:3]
    
    z_min_local = 0  ###np.min(ag.positions[:, 2])
    z_max_local = Lz ###np.max(ag.positions[:, 2])
    print("z_max_local:",z_max_local)
    print("Local cylinder length:",z_max_local-z_min_local)

    for segment in range(num_segments_global):
        segment_z_min = z_min_global + segment * segment_width
        #print(segment_z_min)
        segment_z_max = min(segment_z_min + segment_width, z_max_global)
        #print(segment_z_max)

        segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")

        if len(segment_ag) == 0:
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            print(f"Segment {segment+1} has no atoms, appended NaN")
            continue  # Skip to next segment

        max_z_in_segment = np.max(segment_ag.positions[:, 2])

        if (max_z_in_segment - segment_z_min) < (segment_width / 2):
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            print(f"Max z too small in segment {segment+1}, appended NaN")
            continue

        center_of_mass = segment_ag.center_of_mass()
        distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1)
        mean_segment_values[f'Segment_{segment+1}'].append(np.mean(distances))
        #print("\n")

# Convert to DataFrame : # Finding Constriction radius = smallest mean radius across segments per frame

mean_segment_df = pd.DataFrame(mean_segment_values)

mean_segment_df_plot = mean_segment_df.iloc[:, :-3]  # Keep all rows, all but last 3 columns
mean_segment_df_plot.to_csv('mean_segment_df_V2.txt', sep='\t', index=False)


filtered_df = mean_segment_df.replace(0, np.nan)  # Just in case zeros still exist
mean_segment_df["constrict_radius"] = filtered_df.min(axis=1, skipna=True)
#print(mean_segment_df)


# Save frame index + constriction radius
frame_radius_pairs = list(zip(mean_segment_df.index, mean_segment_df["constrict_radius"]))
with open("frame_radius_pairs.txt", "w") as f:
    for frame, radius in frame_radius_pairs:
        f.write(f"{frame}\t{radius}\n")



# Drop NaNs for histogram
data = mean_segment_df["constrict_radius"].dropna()
data.to_csv("constriction_radius.txt", index=False, header=False)

# Fit Gaussian
mu, sigma = norm.fit(data)

# Plot histogram
count, bins, ignored = plt.hist(data, bins=30, alpha=0.6, color='skyblue', edgecolor='black')
x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, mu, sigma) * len(data) * (bins[1] - bins[0])
plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fit: μ={mu:.2f}, σ={sigma:.2f}')
plt.axvline(mu - sigma, color='orange', linestyle='--', linewidth=2)
plt.axvline(mu + sigma, color='orange', linestyle='--', linewidth=2)
plt.axvline(mu, color='blue', linestyle='--', linewidth=2)
plt.xlabel("Constriction Radius (Å)")
plt.ylabel("Count")
plt.legend()
plt.savefig("gaussian_constrict_radius.png", dpi=300, bbox_inches="tight")
plt.show()

# Assign frames to bins : This code groups frame indices into bins based on their radius values.
# Each bin contains the frames where the radius falls within that bin's range.
# NaNs are ignored. 
# The mapping is stored in bin_frame_map.

bin_frame_map = {i: [] for i in range(len(bins)-1)} #{0: [], 1: [], ..., N-1: []} each key is a bin index, and value will be a list of frame indices falling into that bin.
#print(bin_frame_map)


for frame_idx, radius in frame_radius_pairs:           #  For each frame:(frame index and the radius)
    #print("frame_idx:",frame_idx)
    #print("radius:",radius)
    if np.isnan(radius):
        continue
    bin_idx = np.digitize(radius, bins) - 1   # returns index of bin into which radius falls. bin indices start at 1, so subtracting 1 makes them start at 0 — match dictionary keys.
    #print(bin_idx)
    #print("\n")
    if 0 <= bin_idx < len(bins)-1:
        bin_frame_map[bin_idx].append(frame_idx)
        #print(f"Frame {frame_idx} falls within bin {bin_idx}")

        
#print(bin_frame_map)

# Write out trajectories for each bin
for bin_idx, frame_list in bin_frame_map.items():
    r_min = bins[bin_idx]
    r_max = bins[bin_idx + 1]
    
    
    if not frame_list:
        filename = f"bin_Empty_{bin_idx}.xtc"
        print(f"Skipping empty bin {bin_idx} → {filename} (radius range: {r_min:.2f}-{r_max:.2f})")
        continue

    filename = f"bin_{bin_idx}.xtc"
    print(f"Writing {filename} with {len(frame_list)} frames (radius range: {r_min:.2f}-{r_max:.2f})")

    # Write only frames belonging to this bin
    with mda.Writer(filename, u.atoms.n_atoms) as W:
        for ts in u.trajectory[frame_list]:
            W.write(u.atoms)

print("Done. Generated:")




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

# Remove last 3 segments
segments_to_plot = list(mean_segment_values.keys())[:-3]
num_segments = len(segments_to_plot)

# Create subplots
fig, axes = plt.subplots(
    nrows=int(np.ceil(num_segments / 3)),
    ncols=3,
    figsize=(15, 10)
)
axes = axes.flatten()
segment_means = []

for i, segment in enumerate(segments_to_plot):
    data = mean_segment_df[segment].dropna()
    mean = data.mean()
    std_dev = data.std()
    segment_means.append(mean)

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





























