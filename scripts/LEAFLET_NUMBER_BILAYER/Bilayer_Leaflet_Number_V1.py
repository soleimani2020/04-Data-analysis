import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm  # for progress bars
import MDAnalysis as mda
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from MDAnalysis.lib.distances import distance_array


import numpy as np
import matplotlib.pyplot as plt

def count_lipids_cylindrical_x(
    universe,
    frame_number,
    segment,
    lipid_resnames=["DLPC"],
    headgroup_bead="PO4",
    tailgroup_bead="C3B",
    cylinder_axis=0,
    neighbor_cutoff=12.0,
    max_iterations=20,
):
    """Count lipids in inner and outer leaflets of a cylindrical membrane,
       treating lipids exactly on the mean radius as inner.
    """
    # Select all lipid atoms
    lipids = universe.select_atoms(f"resname {' '.join(lipid_resnames)}")
    print(f"Total lipid molecules: {len(lipids)/10}")
    
    # Select headgroup and tailgroup beads
    headgroups = lipids.select_atoms(f"name {headgroup_bead}")
    print(f"Headgroup beads found: {len(headgroups)}")
    
    tailgroups = lipids.select_atoms(f"name {tailgroup_bead}")
    print(f"Tailgroup beads found: {len(tailgroups)}")
    
    # Coordinates in the XY plane (perpendicular to cylinder axis)
    coords = headgroups.positions[:, [0,1,2]]
    coords_tailgroup = tailgroups.positions[:, [0,1,2]]
    
    # Compute center of mass
    segment_ag = lipids.select_atoms(f"prop z >= {-1000000} and prop z < {1000000}")
    center_of_mass = segment_ag.center_of_mass()
    print("Center of mass:", center_of_mass)
    
    # Compute radial distances from COM in XY plane
    distances = np.linalg.norm(coords[:, 0:2] - center_of_mass[0:2], axis=1)
    distances_tails = np.linalg.norm(coords_tailgroup[:, 0:2] - center_of_mass[0:2], axis=1)
    mean_radius = np.median(distances_tails)
    print(f"Membrane median radius: {mean_radius:.2f} Å\n")
    
    # Classification: inner <= mean_radius, outer > mean_radius
    inner_mask = distances <= mean_radius
    outer_mask = distances > mean_radius
    
    inner_indices = np.where(inner_mask)[0]
    outer_indices = np.where(outer_mask)[0]
    
    # Count lipids exactly on the border
    border_indices = np.where(np.isclose(distances, mean_radius, atol=1e-6))[0]
    print(f"Lipids exactly on the border (counted as inner): {len(border_indices)}")
    
    R_inner_PO4 = np.mean(distances[inner_indices])
    R_outer_PO4 = np.mean(distances[outer_indices])
    
    print(f"Average inner radius (PO4): {R_inner_PO4:.3f} Å")
    print(f"Average outer radius (PO4): {R_outer_PO4:.3f} Å")
    
    # ---------------------------
    # 3D scatter + XY circle at z = COM_z
    # ---------------------------
    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    colors = np.zeros(len(headgroups))
    colors[outer_indices] = 1
    
    # COM
    ax3d.scatter(*center_of_mass, color='green', s=50, label='Center of Mass')
    
    # Headgroups
    ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=colors, cmap='bwr', s=10, alpha=0.6)
    
    # Circle at mean radius
    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = center_of_mass[0] + mean_radius * np.cos(theta)
    circle_y = center_of_mass[1] + mean_radius * np.sin(theta)
    circle_z = np.full_like(circle_x, fill_value=center_of_mass[2])
    ax3d.plot(circle_x, circle_y, circle_z, color='green', linestyle='--', linewidth=3, label='Mean Radius Circle (XY)')
    
    ax3d.set_title('Final Classification (Red=Outer, Blue=Inner)')
    fig.colorbar(ax3d.collections[0], ax=ax3d, shrink=0.5, label='Outer (1) / Inner (0)')
    
    # ---------------------------
    # 2D XY top-down view
    # ---------------------------
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(center_of_mass[0], center_of_mass[1], color='green', s=100, label='Center of Mass', zorder=3)
    ax.scatter(coords[:,0], coords[:,1], c=colors, cmap='bwr', s=10, alpha=0.6, zorder=2)
    ax.plot(circle_x, circle_y, color='green', linestyle='--', linewidth=2, label='Mean Radius Circle', zorder=1)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title('2D X–Y View (Red=Outer, Blue=Inner)')
    ax.legend()
    ax.axis('equal')
    fig2.tight_layout()

    plt.show()
    
    return len(inner_indices), len(outer_indices), inner_indices, outer_indices, R_inner_PO4, R_outer_PO4





####### 
import MDAnalysis as mda
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load universe for initial setup
universe = mda.Universe("nvt_new.tpr", "traj_comp.xtc")
lipid_resnames = ["DLPC"]
ag = universe.select_atoms(f"resname {' '.join(lipid_resnames)}")

print("Number of lipid atoms:", len(ag))

cylinder_x_min = 0#np.min(ag.positions[:, 2])
cylinder_x_max = np.max(ag.positions[:, 2])+50
print("Cylinder length (A):", cylinder_x_max - cylinder_x_min)

segment_width = 100  # Width of each segment in Angstroms
total_length = cylinder_x_max - cylinder_x_min
num_segments = int(np.ceil(total_length / segment_width))  # Ensure full coverage
print(num_segments)



def analyze_first_10_frames():
    # Load the universe
    uni = mda.Universe("nvt_new.tpr", "traj_comp.xtc")
    ag = uni.select_atoms(f"resname {' '.join(lipid_resnames)}")
    print("TOTALLL:",len(ag.positions))
    
    
    # Limit to first 10 frames
    n_frames = min(10, len(uni.trajectory))
    
    # Store results here
    results = []
    
    for ts in tqdm(uni.trajectory[:n_frames], total=n_frames):
    #for ts in tqdm(uni.trajectory[6:7], total=n_frames):
        print(f"Processing Frame {ts.frame}")
        frame_data = []

        for segment in range(num_segments):
            segment_x_min = cylinder_x_min + segment * segment_width
            print("segment_x_min:\n",segment_x_min)

            # Fix the last segment to go exactly to cylinder_x_max
            if segment == num_segments - 1:
                segment_x_max = cylinder_x_max
                print("segment_x_max:\n",segment_x_max)
            else:
                segment_x_max = segment_x_min + segment_width
                print("segment_x_max:\n",segment_x_max)

            segment_ag = ag.select_atoms(f"prop z >= {segment_x_min} and prop z < {segment_x_max}")
            center_of_mass = segment_ag.center_of_mass()

            inner, outer, inner_idx, outer_idx , R_inner_PO4 , R_outer_PO4 = count_lipids_cylindrical_x(segment_ag, ts.frame,segment)

            frame_data.append({
                "Frame": ts.frame,
                "Segment": segment,
                "Inner": inner,
                "Outer": outer,
                "R_Inner":R_inner_PO4,
                "R_Outer":R_outer_PO4,
            })

        # Add segment-wise data to the main list
        results.extend(frame_data)

        # Compute stats
        # Compute stats
        inner_vals = [row["Inner"] for row in frame_data]
        outer_vals = [row["Outer"] for row in frame_data]
        R_inner_vals = [row["R_Inner"] for row in frame_data]
        R_outer_vals = [row["R_Outer"] for row in frame_data]

        avg_inner = np.mean(inner_vals)
        avg_outer = np.mean(outer_vals)
        sum_inner = np.sum(inner_vals)
        sum_outer = np.sum(outer_vals)
        avg_R_inner = np.mean(R_inner_vals)
        avg_R_outer = np.mean(R_outer_vals)

        # Add average row
        results.append({
            "Frame": ts.frame,
            "Segment": "Average",
            "Inner": round(avg_inner, 2),
            "Outer": round(avg_outer, 2),
            "R_Inner": round(avg_R_inner, 3),
            "R_Outer": round(avg_R_outer, 3)
        })

        # Add summation row
        results.append({
            "Frame": ts.frame,
            "Segment": "Sum",
            "Inner": int(sum_inner),
            "Outer": int(sum_outer),
            "R_Inner": "",
            "R_Outer": ""
        })

        # Blank row for spacing
        results.append({
            "Frame": "",
            "Segment": "",
            "Inner": "",
            "Outer": "",
            "R_Inner": "",
            "R_Outer": ""
        })


    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Write to file
    df.to_csv("inner_outer_counts.txt", sep='\t', index=False)
    print("Results saved to inner_outer_counts.txt")
    
    
    # Write full results
    df.to_csv("inner_outer_counts.txt", sep='\t', index=False)
    print("Results saved to inner_outer_counts.txt")

    # === NEW: only keep SUM per frame ===
    df_sum = df[df["Segment"] == "Sum"][["Frame", "Inner", "Outer"]]
    df_sum.to_csv("inner_outer_only.txt", sep='\t', index=False)
    print("Simplified results (per-frame sums) saved to inner_outer_only.txt")
    
    # === Plot Inner & Outer vs Frame ===
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    # Plot inner and outer as lines
    plt.plot(df_sum["Frame"], df_sum["Inner"], '-o', label="Inner", color='blue')
    plt.plot(df_sum["Frame"], df_sum["Outer"], '-s', label="Outer", color='red')

    # Optionally, add a smooth trend line using a rolling mean
    # df_sum['Inner_smooth'] = df_sum['Inner'].rolling(window=3, min_periods=1).mean()
    # df_sum['Outer_smooth'] = df_sum['Outer'].rolling(window=3, min_periods=1).mean()
    # plt.plot(df_sum["Frame"], df_sum['Inner_smooth'], '--', color='blue', alpha=0.7, label='Inner trend')
    # plt.plot(df_sum["Frame"], df_sum['Outer_smooth'], '--', color='red', alpha=0.7, label='Outer trend')

    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.title("Inner and Outer Lipid Counts per Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inner_outer_vs_frame.png", dpi=300)
    plt.show()
    
    
    import matplotlib.pyplot as plt

    # ------------------------
    # Inner lipids vs Frame
    # ------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df_sum["Frame"], df_sum["Inner"], '-o', color='blue', label='Inner')
    # Optional: add smooth trend
    # df_sum['Inner_smooth'] = df_sum['Inner'].rolling(window=3, min_periods=1).mean()
    # plt.plot(df_sum["Frame"], df_sum['Inner_smooth'], '--', color='blue', alpha=0.7, label='Inner trend')
    plt.xlabel("Frame")
    plt.ylabel("Inner Lipid Count")
    plt.title("Inner Lipid Counts per Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inner_vs_frame.png", dpi=300)
    plt.show()

    # ------------------------
    # Outer lipids vs Frame
    # ------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df_sum["Frame"], df_sum["Outer"], '-s', color='red', label='Outer')
    # Optional: add smooth trend
    # df_sum['Outer_smooth'] = df_sum['Outer'].rolling(window=3, min_periods=1).mean()
    # plt.plot(df_sum["Frame"], df_sum['Outer_smooth'], '--', color='red', alpha=0.7, label='Outer trend')
    plt.xlabel("Frame")
    plt.ylabel("Outer Lipid Count")
    plt.title("Outer Lipid Counts per Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outer_vs_frame.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    analyze_first_10_frames()

    
