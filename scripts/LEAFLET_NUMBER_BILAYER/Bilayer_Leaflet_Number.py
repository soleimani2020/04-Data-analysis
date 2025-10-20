import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm  # for progress bars
import MDAnalysis as mda
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from MDAnalysis.lib.distances import distance_array




def count_lipids_cylindrical_x(universe, frame_number, segment,
                                lipid_resnames=["DLPC"], headgroup_bead="PO4", tailgroup_bead="C3B",
                                cylinder_axis=0, neighbor_cutoff=12.0, max_iterations=20):
    """
    Count lipids above and below the median Z position of the tailgroup COM.
    """

    # Output file base name
    output_file = "Leaflet_Classification"

    # Select all lipid atoms
    lipids = universe.select_atoms(f"resname {' '.join(lipid_resnames)}")
    print(f"Total lipid molecules: {len(lipids) / 10}")

    # Select headgroup and tailgroup beads
    headgroups = lipids.select_atoms(f"name {headgroup_bead}")
    tailgroups = lipids.select_atoms(f"name {tailgroup_bead}")
    print(f"Headgroup beads found: {len(headgroups)}")
    print(f"Tailgroup beads found: {len(tailgroups)}")

    # Get coordinates
    coords = headgroups.positions  # Headgroup positions (N, 3)
    coords_tailgroup = tailgroups.positions  # Tailgroup positions (M, 3)

    # Compute full COM of tailgroup
    center_of_mass = coords_tailgroup.mean(axis=0)  # [X, Y, Z]
    z_com_tailgroup = center_of_mass[2]

    # Classify headgroups by their Z-position relative to COM Z
    z_coords = coords[:, 2]
    outer_mask = z_coords > z_com_tailgroup
    outer_indices = np.where(outer_mask)[0]
    inner_indices = np.where(~outer_mask)[0]

    print(f"Membrane Z COM (tailgroup): {z_com_tailgroup:.2f} Å")
    print(f"Headgroups above COM: {len(outer_indices)}")
    print(f"Headgroups below COM: {len(inner_indices)}")
    
    
    # --- Write to text file ---

    import os  # Add this at the top of your script

    # --- Logging block ---
    log_filename = f"{output_file}_counts.txt"

    # Write header only once (if file does not exist yet)
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as f:
            f.write("frame_number  inner_count  outer_count\n")

    # Append current frame data
    with open(log_filename, "a") as f:
        f.write(f"{frame_number}  {len(inner_indices)}  {len(outer_indices)}\n")

        

    # --- 3D Visualization ---
    # fig = plt.figure(figsize=(10, 6))
    # ax2 = fig.add_axes([0, 0, 1, 1], projection='3d')
    # 
    # # Color classification: red = below (inner), blue = above (outer)
    # colors = np.zeros(len(coords))
    # colors[outer_indices] = 1
    # 
    # # Plot tailgroup COM
    # ax2.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2],
    #             color='green', s=50, label='Tailgroup COM (Z threshold)')
    # 
    # # Plot headgroup positions
    # sc2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
    #                   c=colors, cmap='bwr', s=10, alpha=0.6)
    # 
    # # Add horizontal Z plane
    # ax2.plot([center_of_mass[0]] * 2, [center_of_mass[1]] * 2,
    #          [z_com_tailgroup - 5, z_com_tailgroup + 5],
    #          color='green', linestyle='--', linewidth=2)
    # 
    # # Labels and visuals
    # ax2.set_title('Z-Direction Classification (Blue = Above, Red = Below)')
    # ax2.set_xlabel('X (Å)')
    # ax2.set_ylabel('Y (Å)')
    # ax2.set_zlabel('Z (Å)')
    # ax2.legend()
    # 
    # fig.colorbar(sc2, ax=ax2, shrink=0.5, label='Above (1) / Below (0)')
    # 
    # #plt.savefig(f"{output_file}_{frame_number}.png", dpi=300, bbox_inches='tight')
    # #plt.tight_layout()
    # #plt.show()

    return len(inner_indices), len(outer_indices), inner_indices, outer_indices



####### 
import MDAnalysis as mda
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load universe for initial setup
universe = mda.Universe("cyl.tpr", "traj_comp0.xtc")
lipid_resnames = ["DLPC"]
ag = universe.select_atoms(f"resname {' '.join(lipid_resnames)}")
print("Number of lipid atoms:", len(ag))

cylinder_x_min = np.min(ag.positions[:, 2])
cylinder_x_max = np.max(ag.positions[:, 2])
print("Membrane thickness (A):", cylinder_x_max - cylinder_x_min)

total_length = cylinder_x_max - cylinder_x_min
num_segments = 1# int(np.ceil(total_length / segment_width))  # Ensure full coverage



def analyze_first_10_frames():
    # Load the universe
    uni = mda.Universe("cyl.tpr", "traj_comp0.xtc")
    ag = uni.select_atoms(f"resname {' '.join(lipid_resnames)}")
    
    # Limit to first 10 frames
    n_frames = min(20, len(uni.trajectory))
    
    # Store results here
    results = []

    for ts in tqdm(uni.trajectory[:n_frames], total=n_frames):
        print(f"Processing Frame {ts.frame}")
        frame_data = []

        for segment in range(1):
            segment_ag = ag.select_atoms(f"prop z >= {-1000000000} and prop z < {1000000000}")
            center_of_mass = segment_ag.center_of_mass()

            inner, outer, inner_idx, outer_idx  = count_lipids_cylindrical_x(segment_ag, ts.frame,segment)

            frame_data.append({
                "Frame": ts.frame,
                "Segment": segment,
                "Inner": inner,
                "Outer": outer,
            })

        # Add segment-wise data to the main list
        results.extend(frame_data)

        # Compute stats
        # Compute stats
        inner_vals = [row["Inner"] for row in frame_data]
        outer_vals = [row["Outer"] for row in frame_data]


        avg_inner = np.mean(inner_vals)
        avg_outer = np.mean(outer_vals)
        sum_inner = np.sum(inner_vals)
        sum_outer = np.sum(outer_vals)


        # Add average row
        results.append({
            "Frame": ts.frame,
            "Segment": "Average",
            "Inner": round(avg_inner, 2),
            "Outer": round(avg_outer, 2),
        })

        # Add summation row
        results.append({
            "Frame": ts.frame,
            "Segment": "Sum",
            "Inner": int(sum_inner),
            "Outer": int(sum_outer),
        })

        # Blank row for spacing
        results.append({
            "Frame": "",
            "Segment": "",
            "Inner": "",
            "Outer": "",
        })


    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Write to file
    df.to_csv("inner_outer_counts.txt", sep='\t', index=False)
    print("Results saved to inner_outer_counts.txt")

if __name__ == "__main__":
    analyze_first_10_frames()


###########


import matplotlib.pyplot as plt

# Replace with your actual filename
filename = "Leaflet_Classification_counts.txt"

# Initialize lists to store data
frame_numbers = []
inner_counts = []
outer_counts = []

# Read file and parse data
with open(filename, "r") as f:
    # Skip header line
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue  # skip malformed lines
        frame, inner, outer = parts
        frame_numbers.append(int(frame))
        inner_counts.append(int(inner))
        outer_counts.append(int(outer))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, inner_counts, label='Inner Count', marker='o')
plt.plot(frame_numbers, outer_counts, label='Outer Count', marker='s')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.title('Inner and Outer Lipid Counts per Frame')
plt.legend()
plt.savefig(f"Result.png", dpi=300, bbox_inches='tight')
plt.show()

