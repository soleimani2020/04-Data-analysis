import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import math 
import numpy as np


# Load system
u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")
Frame_Number  = 50
Bin_Number    = 50   # bins of histogram
num_bins = Bin_Number
segment_width = 20  # Angstroms ( To calculate the radius )

# First pass: Determine global min/max z
z_min_global = np.inf
z_max_global = -np.inf


### Finding maximun system size 

for ts in u.trajectory[:10]:
    # Select all tail beads
    ag = u.select_atoms('name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B')
    z_min_global = 0      #min(z_min_global, np.min(ag.positions[:, 2]))
    z_max_global = 788.17 #max(z_max_global, np.max(ag.positions[:, 2]))

cylinder_length_global =  z_max_global - z_min_global
num_segments_global = int(np.ceil(cylinder_length_global / segment_width)) # np.ceil() rounds that number up to the nearest integer
print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
print(f"Fixed number of segments: {num_segments_global}")

# Initialize storage for segment stats
mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}
frame_radius_pairs = []





# Iterate over frames for Finding maximum and minimum radius 

mean_segment_values["z_max_local"] = []  # initialize list

for ts in u.trajectory[:Frame_Number]:
    print(f"Processing Frame {ts.frame}")
    ag = u.select_atoms('name C3B')
    box = ts.dimensions 
    Lx, Ly, Lz = box[:3]
    z_min_local = 0
    z_max_local = Lz
    mean_segment_values["z_max_local"].append(z_max_local)  # add z_max_local for this frame

    for segment in range(num_segments_global):
        segment_z_min = z_min_global + segment * segment_width
        segment_z_max = min(segment_z_min + segment_width, z_max_global)

        segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")

        if len(segment_ag) == 0:
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            continue

        max_z_in_segment = np.max(segment_ag.positions[:, 2])

        if (max_z_in_segment - segment_z_min) < (segment_width / 2):
            mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
            continue

        center_of_mass = segment_ag.center_of_mass()
        distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1)
        mean_segment_values[f'Segment_{segment+1}'].append(np.mean(distances))


mean_segment_df = pd.DataFrame(mean_segment_values)


# Get only the segment columns (exclude z_max_local)
segment_cols = [col for col in mean_segment_df.columns if col.startswith("Segment_")]
filtered_segments_df = mean_segment_df[segment_cols].replace(0, np.nan)
# Most Constriction radius (min of segments)
mean_segment_df["constrict_radius"] = filtered_segments_df.min(axis=1, skipna=True)
# Least Constriction radius (max of segments)
mean_segment_df["max_radius"] = filtered_segments_df.max(axis=1, skipna=True)
mean_segment_df.to_csv("mean_segment_df.csv", index=False)
print("mean_segment_df.head():\n", mean_segment_df)

global_min = filtered_segments_df.min().min()
global_max = filtered_segments_df.max().max()

########################################################################################################
########################################################################################################
########################################################################################################
print(f"###################  Min and Max radius #######################################")
print("Global min:", global_min)
print("Global max:", global_max)
print(f"###################  Histogram Bin Size #######################################")
bin_Size = (global_max-global_min)/Bin_Number
print("Bin Size:\n",bin_Size)
print(f"###################  Histogram Bin Number #######################################")
print("Bin_Number:\n",Bin_Number)    
print(f"###################  Histogram Bins  #######################################")
bins = np.linspace(global_min, global_max, Bin_Number+1)
print("BINS:",bins)
print(f"######################################################################################")

z_positions = np.arange(num_segments_global) * segment_width  # 0 is Begining of the first segment ,  last nuumer is the begining of the last segment
#print("z_positions:",z_positions)
print("\n")


shifted_data_rows = []

for bin_index in range(Bin_Number):
#for bin_index in range(1):
    R_min_bin = bins[bin_index]
    R_max_bin = bins[bin_index + 1]

    for frame_index, row in mean_segment_df.iterrows():
            
            
        Actual_system_length_z = row["z_max_local"]
        system_length_z  = row["z_max_local"]
        #print("Actual system_length_z:", system_length_z)
            

        for seg_index in range(num_segments_global):
            radius = row[f'Segment_{seg_index+1}']
            
            if np.isnan(radius):
                continue
            
            if R_min_bin <= radius < R_max_bin:
                #print("seg_index, R_min_bin, radius, R_max_bin:",seg_index+1, R_min_bin, radius, R_max_bin)
                
                # Segment centers along z
                z_positions_old = np.arange(num_segments_global) * segment_width 
                
                z_positions = np.arange(num_segments_global) * segment_width 
                #print("z_positions old:\n",z_positions)
                
                valid_indices = z_positions < system_length_z
                z_positions = z_positions[valid_indices]
                
                

                if (Actual_system_length_z - z_positions[-1]) < (segment_width / 2): 
                        system_length_z =  z_positions[-1]   # Truncate
                        #print("1.Round system_length_z:",system_length_z)
                else:
                    system_length_z =  z_positions[-1] + segment_width  # Extend
                    #print("2.Round system_length_z:",system_length_z)

                
                
                valid_indices = z_positions < system_length_z
                z_positions = z_positions[valid_indices]
                #print("z_positions new:\n",z_positions)
                
                
                # Also filter the corresponding radii
                radii_to_use = []
                for i, valid in enumerate(valid_indices):
                    if valid:
                        radii_to_use.append(row[f'Segment_{i+1}'])
                radii_to_use = np.array(radii_to_use)
                #print("radii_to_use new:\n",radii_to_use)

                # 🚨 Skip this seg_index if it’s outside the valid range
                if not valid_indices[seg_index]:
                    continue
                
                
                
                                
                ########### To do ! ###############
                
                # shift current segment to zero
                shifted_z = z_positions - z_positions[np.where(valid_indices)[0].tolist().index(seg_index)]
                #print("shifted_z:",shifted_z)
                
                # Wrap (Only) negatives using modulo operation
                shifted_z_periodic = np.mod(shifted_z, system_length_z)
                #print("shifted_z_periodic:",shifted_z_periodic)
                
                
                ##################################
                
                
                

                ## DOUBLE CHECK PBC VISUALLY
#                 z_positions = np.array(z_positions_old)
#                 radii_to_use = np.array(radii_to_use)
#                 
#                 # Ensure both arrays have same length
#                 min_len = min(len(z_positions), len(radii_to_use))
#                 z_positions = z_positions[:min_len]
#                 #print("z_positions:",z_positions)
#                 radii_to_use = radii_to_use[:min_len]
#                 #print("radii_to_use:",radii_to_use)
#                 
#                 # Filter out NaNs in radii
#                 valid_mask = ~np.isnan(radii_to_use)
#                 z_positions_filtered = z_positions[valid_mask]
#                 radii_filtered = radii_to_use[valid_mask]
#                 
#                 
#                 # Create figure
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(z_positions_filtered, radii_filtered, 'o-', color='blue', markersize=6, linewidth=2, alpha=0.7)
#                 
#                 # Annotate points
#                 for z, r in zip(z_positions_filtered, radii_filtered):
#                     plt.text(z, r + 0.2, f'{r:.2f}', fontsize=9, ha='center', va='bottom')
#                 
#                 # Labels and title
#                 plt.xlabel('Z Position (Å)', fontsize=14)
#                 plt.ylabel('Radius (Å)', fontsize=14)
#                 plt.title('Radius vs. Z Position', fontsize=16)
#                 plt.grid(True, alpha=0.3)
#                 plt.ylim(40, 60)
#                 
#                 # Reference lines
#                 plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
#                 plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
#                 
#                 # Save and show
#                 plt.savefig(f'radius_vs_z_positions_row{frame_index}.png', dpi=300, bbox_inches='tight')
#                 print("Figure saved as 'radius_vs_z_positions.png'")
#                 plt.show()
#                 plt.close()  # Free memory



                

                # Store a row as a dictionary
                row_dict = {
                    "R_min_bin": R_min_bin,
                    "radius": radius,
                    "R_max_bin": R_max_bin,
                    "system_length_z": system_length_z,
                    "bin_index": bin_index, 
                    "frame_index": frame_index, 
                    "segment_index": seg_index+1,
                }
                
                for i in range(len(shifted_z)):
                    row_dict[f"zshifted_{i}"] = shifted_z[i]
                    row_dict[f"zperiodic_{i}"] = shifted_z_periodic[i]
                    row_dict[f"r_{i}"] = row[f"Segment_{i+1}"]
                
                shifted_data_rows.append(row_dict)

# Convert to DataFrame
shifted_df = pd.DataFrame(shifted_data_rows)
print("shifted_df.head():\n", shifted_df.head())
shifted_df.to_csv("shifted_df.csv", index=False)


# -------------------------
# 2️⃣ Plot Average
# -------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create position range from 0 to 800 Å 
positions = np.arange(0, 801, segment_width)

results = []

# First, find all unique bin indices in your data
unique_bins = sorted(shifted_df['bin_index'].unique())
print(f"##########################################################")
print(f"Found {len(unique_bins)} unique bins: {unique_bins}")
print(f"##########################################################")

# Process each bin
for bin_index in unique_bins:
    df_bin = shifted_df[shifted_df["bin_index"] == bin_index]
    for position in positions:
        all_radii = []
        for seg_index in range(num_segments_global):  
            zshifted_col = f"zshifted_{seg_index}"
            zperiodic_col = f"zperiodic_{seg_index}"
            radius_col = f"r_{seg_index}"
            
            # Check if these columns exist in the DataFrame
            if zshifted_col in df_bin.columns and zperiodic_col in df_bin.columns and radius_col in df_bin.columns:
                # Exact matching (no tolerance), including NaNs
                zshifted_matches = (df_bin[zshifted_col] == position) | (df_bin[zshifted_col].isna() & pd.isna(position))
                zperiodic_matches = (df_bin[zperiodic_col] == position) | (df_bin[zperiodic_col].isna() & pd.isna(position))
                
                # Combine matches
                combined_matches = zshifted_matches | zperiodic_matches
                
                # Get the radius values for matches
                radii_at_position = df_bin.loc[combined_matches, radius_col].dropna().tolist()
                all_radii.extend(radii_at_position)
        
        # Calculate average radius if there are measurements
        if len(all_radii) > 0:
            average_radius = np.mean(all_radii)
        else:
            average_radius = np.nan  # Use NaN if no measurements found
        
        results.append({
            "bin_index": bin_index,
            "position_Å": position,
            "radius_measurements": all_radii,
            "measurement_count": len(all_radii),
            "average_radius": average_radius
        })

# Create DataFrame
result_df = pd.DataFrame(results)
print("result_df:\n",result_df)  
result_df.to_csv("radius_measurements_by_position.csv", index=False)

# Create a separate plot for each bin
for bin_index in unique_bins:
    bin_data = result_df[result_df['bin_index'] == bin_index]
    plot_df = bin_data.dropna(subset=['average_radius'])
    
    if len(plot_df) > 0:
        plt.figure(figsize=(12, 8))
        plt.plot(plot_df['position_Å'], plot_df['average_radius'], 
                 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)

        # Annotate each point with its value
        for x, y in zip(plot_df['position_Å'], plot_df['average_radius']):
            plt.text(x, y + 0.2, f'{y:.2f}', fontsize=9, ha='center', va='bottom')

        plt.xlabel('Position (Å)', fontsize=14)
        plt.ylabel('Average Radius (Å)', fontsize=14)
        plt.title(f'Average Radius vs. Position along the cylinder (Bin {bin_index})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
        plt.ylim(40, 60)

        plt.savefig(f'average_radius_vs_position_bin_{bin_index}.png', dpi=300, bbox_inches='tight')
        print(f"Plot for bin {bin_index} saved! Found {len(plot_df)} positions with valid radius measurements.")
        plt.close()
    else:
        print(f"No valid measurements found for bin {bin_index}")

# Combined plot with all bins
plt.figure(figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bins)))

for i, bin_index in enumerate(unique_bins):
    bin_data = result_df[result_df['bin_index'] == bin_index]
    plot_df = bin_data.dropna(subset=['average_radius'])
    
    if len(plot_df) > 0:
        plt.plot(plot_df['position_Å'], plot_df['average_radius'], 
                 'o-', linewidth=2, markersize=4, color=colors[i], 
                 alpha=0.7, label=f'Bin {bin_index}')
        
        # Annotate each point
        for x, y in zip(plot_df['position_Å'], plot_df['average_radius']):
            plt.text(x, y + 0.2, f'{y:.2f}', fontsize=8, ha='center', va='bottom', color=colors[i])

plt.xlabel('Position (Å)', fontsize=14)
plt.ylabel('Average Radius (Å)', fontsize=14)
plt.title('Average Radius vs. Position along the cylinder (All Bins)', fontsize=16)
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
plt.ylim(40, 60)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('average_radius_vs_position_all_bins.png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots created successfully!")




