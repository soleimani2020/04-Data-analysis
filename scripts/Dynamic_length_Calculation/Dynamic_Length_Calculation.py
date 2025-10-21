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

num_segments_global = 37 

# Load system
u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")


# First pass: Determine global min/max z
z_min_global = np.inf
z_max_global = -np.inf

data = []
FRAME_ANALYSIS=10000

for ts in u.trajectory[:FRAME_ANALYSIS]:
    ag = u.select_atoms('name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B')
    z_min_global = min(z_min_global, np.min(ag.positions[:, 2]))
    z_max_global = max(z_max_global, np.max(ag.positions[:, 2]))
    z_max_frame = np.max(ag.positions[:, 2])  # max z in THIS frame
    data.append({"frame": ts.frame, "z_max": z_max_frame})


# Create DataFrame
df = pd.DataFrame(data)
# Sort by z_max descending
df_sorted = df.sort_values(by="z_max", ascending=False)
# Assume df is already created from your data
max_z = df['z_max'].max()
min_z = df['z_max'].min()
difference = max_z - min_z
#print(f"Difference between longest and smallest z_max: {difference:.2f} Å")



cylinder_length_global = z_max_global - z_min_global
print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
print(f"Fixed number of segments: {num_segments_global}")

# Initialize storage for segment stats
mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}

# Store frame index + constriction radius for later binning
frame_radius_pairs = []

# Initialize min/max segment width
min_segment_width = np.inf
max_segment_width = -np.inf



# Iterate over frames
for ts in u.trajectory[:FRAME_ANALYSIS]:
    print(f"Processing Frame {ts.frame}")
    ag = u.select_atoms('name C3B')
    
    if len(ag) == 0:
        continue  # skip frames with no atoms
    
    box = ts.dimensions 
    Lx, Ly, Lz = box[:3]
    
    # Get local z min/max for this frame
    z_min_local = 0  # np.min(ag.positions[:, 2])
    z_max_local = Lz # np.max(ag.positions[:, 2])
    Length_local = z_max_local - z_min_local 
    #print("Length_local:\n",Length_local)
    
    
    # Dynamic segment width
    segment_width = Length_local / num_segments_global   
    
    # Update min/max segment width
    min_segment_width = min(min_segment_width, segment_width)
    max_segment_width = max(max_segment_width, segment_width)
    
    

    # Loop over segments
    for segment in range(num_segments_global):
        segment_z_min = z_min_local + segment * segment_width
        #print(segment_z_min)
        segment_z_max = segment_z_min + segment_width
        #print(segment_z_max)

        segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")

        if len(segment_ag) == 0:
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            #print(f"Segment {segment+1} has no atoms, appended NaN")
            continue  # skip the rest of the current loop iteration and move on to the next segment,


        ### NOT NECESSARY
        # max_z_in_segment = np.max(segment_ag.positions[:, 2])
        # if (max_z_in_segment - segment_z_min) < (segment_width / 2):
        #     mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
        #     print(f"Max z too small in segment {segment+1}, appended NaN")
        #     continue

        center_of_mass = segment_ag.center_of_mass()
        distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1)
        mean_segment_values[f'Segment_{segment+1}'].append(np.mean(distances))
        


# After all frames
print(f"Minimum segment width across all frames: {min_segment_width:.3f}")
print(f"Maximum segment width across all frames: {max_segment_width:.3f}")


# Convert to DataFrame 
mean_segment_df = pd.DataFrame(mean_segment_values)
mean_segment_df.to_csv('mean_segment_df_V2.txt', sep='\t', index=False)
mean_segment_df.to_csv('mean_segment_df_V2.csv', index=False)

# Count number of samples in each column
samples_per_column = mean_segment_df.count()

# Write counts to a text file
with open('samples_per_column.txt', 'w') as f:
    f.write(f"{min_segment_width}\t{min_segment_width}\n")
    f.write(f"{max_segment_width}\t{max_segment_width}\n")
    f.write("Column\tNumSamples\n")
    for col, count in samples_per_column.items():
        f.write(f"{col}\t{count}\n")
        
        
        

mean_segment_df["row_mean"] = mean_segment_df.mean(axis=1)
mean_segment_df.to_csv("mean_segment_df_V2_average.txt", sep="\t", index=False)
mean_segment_df.to_csv("mean_segment_df_V2_average.csv", index=False)



# Compute the average of row_mean
average_radius = mean_segment_df["row_mean"].mean()
# Save it to a text file
with open("average_radius.txt", "w") as f:
    f.write(f"average radius: {average_radius}\n")



######################################################################################################

print("\n\n\n")

segments_to_plot = list(mean_segment_values.keys())
num_segments = len(segments_to_plot)

# Create subplots
fig, axes = plt.subplots(
    nrows=int(np.ceil(num_segments / 3)),
    ncols=3,
    figsize=(15, 10)
)
axes = axes.flatten()
segment_means = []
segment_std_devs = []  # store standard deviations

for i, segment in enumerate(segments_to_plot):
    data = mean_segment_df[segment].dropna()
    #data = mean_segment_df[segment]
    #print("data:\n",data)
    mean = data.mean()
    #print("mean:\n",mean)
    std_dev = data.std()
    segment_means.append(mean)
    segment_std_devs.append((segment, std_dev))  # save segment name and std_dev

    axes[i].hist(data, bins=20, density=True, alpha=0.6, color='gray', label='Histogram')
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
    y = norm.pdf(x, mean, std_dev)
    axes[i].plot(x, y, color='blue')
    axes[i].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
    axes[i].axvline(mean + std_dev, color='green', linestyle='--', label=f'St. Dev.: {std_dev:.3f}')
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

# Save standard deviations to text file
with open("segment_std_devs.txt", "w") as f:
    for seg, std in segment_std_devs:
        f.write(f"{seg}: {std:.4f}\n")

# --- Plot mean vs. segment index with error bars ---
plt.figure(figsize=(8, 5))

# Extract just the std_dev values (in the same order as segment_means)
std_devs = [std for _, std in segment_std_devs]

plt.errorbar(
    range(1, num_segments + 1),
    segment_means,
    yerr=std_devs,
    fmt='o-',            # circle markers with connecting line
    capsize=4,           # adds little caps to error bars
    ecolor='gray',       # error bar color
    elinewidth=1         # error bar line width
)

plt.xlabel("Segment Index")
plt.ylabel("Mean Radius (Å)")
plt.ylim(30, 60)  # Set y-axis limits
plt.title("Mean Radius per Segment")
plt.savefig("Radius_Mean_VS_SegNumber.png", dpi=300)
plt.close()


print(" \n The guassian distribution of radius for each segment is generated for the entire trajectory!\n" )
print(" The mean value of the rafius for each frame is calculated as follows:\n" )
print(" 1. Center of mass of all C3B particles is calculated.\n")
print(" 2. Distance of each C3B particle is caculated with respect to the COM excluding x dimention.\n")
print(" 3. Mean of the distance values is represented as the radius of the system.\n")

plt.tight_layout()





























