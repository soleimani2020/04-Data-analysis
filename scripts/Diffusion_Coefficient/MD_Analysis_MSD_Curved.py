import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# --- Load trajectory ---
# First : gmx trjconv -s nvt_auto.tpr -f traj_first1000 -o traj_first1000_unwrapped.xtc -pbc nojump
# “Unwrapping” reconstructs the continuous trajectory of molecules, ignoring the artificial jumps caused by PBCs
u = mda.Universe("nvt_auto.tpr", "traj_first1000_unwrapped.xtc")  # positions in Å

# Select atoms of interest (lipids)
ag = u.select_atoms("name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B")
residues = ag.residues
residue_group = ag.residues
n_residues = len(residues)
n_frames = len(u.trajectory)
print("Number of Frames:",n_frames)

# --- Arrays to store cylindrical coordinates (per residue COM) ---
coords_R = np.zeros((n_frames, n_residues))
coords_theta = np.zeros((n_frames, n_residues))
coords_z = np.zeros((n_frames, n_residues))


# --- Function to convert positions to cylindrical coordinates ---
def cylindrical_coords(positions, center):
    shifted = positions - center
    x_nm = shifted[:, 0] * 0.1  # Å -> nm
    y_nm = shifted[:, 1] * 0.1
    z_nm = shifted[:, 2] * 0.1
    R = np.sqrt(x_nm**2 + y_nm**2)
    theta = np.arctan2(y_nm, x_nm)
    return R, theta, z_nm

# --- Extract cylindrical coordinates per frame ---
for i, ts in enumerate(u.trajectory):
    # Compute COM per residue
    res_coms = np.array([res.atoms.center_of_mass() for res in residues])
    # Use average COM as cylinder center
    cylinder_center = np.mean(res_coms, axis=0)
    # we get the COM of each reside and then calculate the MSD
    R, theta, z = cylindrical_coords(res_coms, cylinder_center)
    coords_R[i, :] = R
    coords_theta[i, :] = theta
    coords_z[i, :] = z

# --- Compute MSD along curved XY surface ---
max_lag = n_frames
msd_surface = np.zeros(max_lag)

for lag in range(1, max_lag):
    disp2 = 0.0
    count = 0
    for t in range(n_frames - lag):
        dtheta = coords_theta[t + lag, :] - coords_theta[t, :]
        # unwrap angular differences
        dtheta = np.mod(dtheta + np.pi, 2*np.pi) - np.pi
        # displacement along curved surface: ds = R * dtheta
        ds2 = (coords_R[t, :] * dtheta)**2
        disp2 += np.sum(ds2)
        count += n_residues
    msd_surface[lag] = disp2 / count

# --- Time array ---
dt = u.trajectory.dt  # ps per frame
time = np.arange(max_lag) * dt

# --- Fit linear regime for diffusion ---
fit_start = int(0.2 * max_lag)
fit_end = int(0.8 * max_lag)

slope, intercept = np.polyfit(time[fit_start:fit_end], msd_surface[fit_start:fit_end], 1)

# Diffusion coefficient along curved surface (nm²/ps)
D_xy = slope / 4.0  # 2D diffusion
D_xy_cm2_s = D_xy * 1e-2  # convert nm²/ps -> cm²/s

# --- Save results to a text file ---
with open("msd_surface_fit_results.txt", "w") as f:
    f.write(f"Average cylinder radius (nm): {np.mean(coords_R):.6f}\n")
    f.write(f"Slope of MSD fit (nm^2/ps): {slope:.6f}\n")
    f.write(f"Intercept of MSD fit (nm^2): {intercept:.6f}\n")
    f.write(f"Surface diffusion coefficient (nm^2/ps): {D_xy:.6f}\n")
    f.write(f"Surface diffusion coefficient (cm^2/s): {D_xy_cm2_s:.6e}\n")


# --- Plot MSD ---
plt.figure(figsize=(7,5))
plt.plot(time, msd_surface, 'b-', label="MSD along curved surface")
plt.plot(time[fit_start:fit_end], slope*time[fit_start:fit_end] + intercept, 'r--', lw=2, label=f"Linear fit\nslope={slope:.3f} nm²/ps\nD_xy={D_xy:.3f} nm²/ps")
plt.xlabel("Lag time (ps)")
plt.ylabel("MSD along curved surface (nm²)")
plt.title("MSD along cylindrical surface")
plt.legend()
plt.tight_layout()
plt.savefig("MSD_curved_surface.png", dpi=300)
plt.show()


# --- Save MSD data ---
output_data = np.column_stack((time, msd_surface))
np.savetxt("msd_curved_surface.xvg", output_data, header="lagtime\tMSD_xy(nm^2)", fmt="%.6f", delimiter="\t")
