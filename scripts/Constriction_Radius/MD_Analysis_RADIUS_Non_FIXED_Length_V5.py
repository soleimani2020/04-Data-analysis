import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import math 


# Load system
u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")

Bin_Number    = 10
segment_width = 100  # Angstroms

# First pass: Determine global min/max z
z_min_global = np.inf
z_max_global = -np.inf


### Finding maximun system size 

for ts in u.trajectory[:500]:
    # Select all tail beads
    ag = u.select_atoms('name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B')
    z_min_global = min(z_min_global, np.min(ag.positions[:, 2]))
    z_max_global = max(z_max_global, np.max(ag.positions[:, 2]))

cylinder_length_global = z_max_global - z_min_global
num_segments_global = int(np.ceil(cylinder_length_global / segment_width)) # np.ceil() rounds that number up to the nearest integer
print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
print(f"Fixed number of segments: {num_segments_global}")

# Initialize storage for segment stats
mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}
frame_radius_pairs = []





# Iterate over frames for Finding maximum and minimum radius 

for ts in u.trajectory[:10]:
    print(f"Processing Frame {ts.frame}")
    ag = u.select_atoms('name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B')
    box = ts.dimensions 
    Lx, Ly, Lz = box[:3]
    z_min_local = 0  ###np.min(ag.positions[:, 2])
    z_max_local = Lz ###np.max(ag.positions[:, 2])
    #print("z_max_local:",z_max_local)
    #print("Local cylinder length:",z_max_local-z_min_local)
    for segment in range(num_segments_global):
        segment_z_min = z_min_global + segment * segment_width
        #print(segment_z_min)
        segment_z_max = min(segment_z_min + segment_width, z_max_global)
        #print(segment_z_max)

        segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")

        if len(segment_ag) == 0:
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            #print(f"Segment {segment+1} has no atoms, appended NaN")
            continue  # Skip to next segment

        max_z_in_segment = np.max(segment_ag.positions[:, 2])

        if (max_z_in_segment - segment_z_min) < (segment_width / 2):
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            #print(f"Max z too small in segment {segment+1}, appended NaN")
            continue

        center_of_mass = segment_ag.center_of_mass()
        distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1) # Radial distance in X-Y 
        mean_segment_values[f'Segment_{segment+1}'].append(np.mean(distances))



mean_segment_df = pd.DataFrame(mean_segment_values)
filtered_df = mean_segment_df.replace(0, np.nan)  # replaces all 0 with NaN so they ignored when computing min/max/mean.
mean_segment_df["constrict_radius"] = filtered_df.min(axis=1, skipna=True)
mean_segment_df["max_radius"] = filtered_df.max(axis=1, skipna=True)
print(mean_segment_df.head())


global_min = filtered_df.min().min()
global_max = filtered_df.max().max()
print("Global min:", global_min)
print("Global max:", global_max)

print("\n")

########################################################################################################
########################################################################################################
########################################################################################################


bin_Size = (global_max-global_min)/Bin_Number
print("Bin Size:\n",bin_Size)


bins = np.linspace(global_min, global_max, Bin_Number+1)
print("BINS:",bins)

z_positions = np.arange(num_segments_global) * segment_width + segment_width/2
print("z_positions:",z_positions)
#print("\n")

# Create an empty list to store rows
shifted_data_rows = []

for bin_index in range(Bin_Number):
    R_min_bin = bins[bin_index]
    #print(R_min_bin)
    R_max_bin = bins[bin_index + 1]
    #print(R_max_bin)

    for frame_index, row in mean_segment_df.iterrows():
        for seg_index in range(num_segments_global):
            radius = row[f'Segment_{seg_index+1}']
            if np.isnan(radius):
                continue
            
            if R_min_bin <= radius < R_max_bin:
                # Segment centers along z
                z_positions = np.arange(num_segments_global) * segment_width + segment_width/2
                shifted_z = z_positions - z_positions[seg_index]  # shift current segment to zero
                row_dict = {"bin_index": bin_index, "frame_index": frame_index, "segment_index": seg_index+1}
                for i, z in enumerate(shifted_z):
                    row_dict[f"z_{i}"] = z
                    row_dict[f"r_{i}"] = row[f"Segment_{i+1}"]
                
                shifted_data_rows.append(row_dict)

# Convert to DataFrame
shifted_df = pd.DataFrame(shifted_data_rows)
print(shifted_df)


import matplotlib.pyplot as plt
import numpy as np
import math

# Number of bins
num_bins = Bin_Number

# -------------------------
# 1️⃣ Plot each bin in a separate figure
# -------------------------
for bin_index in range(num_bins):
    df_bin = shifted_df[shifted_df["bin_index"] == bin_index]
    
    plt.figure(figsize=(6,5))
    for _, row in df_bin.iterrows():
        shifted_z_cols = [col for col in shifted_df.columns if col.startswith("z_")]
        radius_cols = [col for col in shifted_df.columns if col.startswith("r_")]

        shifted_z = row[shifted_z_cols].values
        radii = row[radius_cols].values

        plt.plot(shifted_z, radii, alpha=0.3, color='blue', marker='o', markersize=3, linewidth=1)
    
    plt.xlabel('Shifted z-position (Å)')
    plt.ylabel('Radius (Å)')
    plt.ylim(40, 60)  # set y-axis limits
    plt.title(f'R(z) curves for bin {bin_index}\nR: {bins[bin_index]:.2f}-{bins[bin_index+1]:.2f} Å')
    plt.tight_layout()
    plt.savefig(f"Rz_bin_{bin_index}.png", dpi=300)
    plt.show()

# -------------------------
# 2️⃣ Plot all bins in one figure
# -------------------------
import matplotlib.pyplot as plt
import math
import numpy as np

# Determine subplot grid size
ncols = 3  # adjust as needed
nrows = math.ceil(num_bins / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), sharex=True, sharey=True)
axes = axes.flatten()

# Colormap for consistency (optional)
colors = plt.cm.viridis(np.linspace(0, 1, num_bins))

for bin_index in range(num_bins):
    ax = axes[bin_index]
    df_bin = shifted_df[shifted_df["bin_index"] == bin_index]

    for _, row in df_bin.iterrows():
        shifted_z_cols = [col for col in df_bin.columns if col.startswith("z_")]
        radius_cols = [col for col in df_bin.columns if col.startswith("r_")]

        shifted_z = row[shifted_z_cols].values
        radii = row[radius_cols].values

        ax.plot(shifted_z, radii, alpha=0.3, color=colors[bin_index], marker='o', markersize=3, linewidth=1)

    ax.set_title(f'Bin {bin_index}\nR: {bins[bin_index]:.2f}-{bins[bin_index+1]:.2f} Å', fontsize=8)
    ax.set_ylim(40, 60)
    ax.set_xlabel('Shifted z (Å)', fontsize=7)
    ax.set_ylabel('Radius (Å)', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)

# Remove empty subplots if any
for i in range(num_bins, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("Rz_bins_subplots.png", dpi=300)
plt.show()







# -------------------------
# 2️⃣ Plot average 
# -------------------------
for bin_index in range(1):
    df_bin = shifted_df[shifted_df["bin_index"] == bin_index]
    print(df_bin)

    if df_bin.empty:
        print(f"Skipping bin {bin_index} (no data)")
        continue

    # get z and radius columns
    shifted_z_cols = [col for col in shifted_df.columns if col.startswith("z_")]
    print(shifted_z_cols)
    radius_cols = [col for col in shifted_df.columns if col.startswith("r_")]
    print(radius_cols)
    # z positions (just take them from the first row, since same for all rows)
    shifted_z = df_bin.iloc[0][shifted_z_cols].values
    print(shifted_z)
    print(df_bin[radius_cols])
    # average radii across rows
    avg_radii = df_bin[radius_cols].mean(axis=0).values
    print(avg_radii)

    # plot the average curve
    plt.figure(figsize=(6,5))
    plt.plot(shifted_z, avg_radii, color='blue', marker='o', markersize=3)
    plt.xlabel('Shifted z-position (Å)')
    plt.ylabel('Average Radius (Å)')
    plt.ylim(40, 60)
    plt.title(f'Average R(z) curve for bin {bin_index}\n'
              f'R: {bins[bin_index]:.2f}-{bins[bin_index+1]:.2f} Å')
    plt.tight_layout()
    plt.savefig(f"Rz_avg_bin_{bin_index}.png", dpi=300)
    plt.show()


