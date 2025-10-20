import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# First unwrap thetrajectory : gmx trjconv -s nvt_auto.tpr -f traj_first1000 -o traj_first1000_unwrapped.xtc -pbc nojump
# Match with : echo "0" | gmx msd -f traj_first1000_unwrapped.xtc -s nvt.tpr -n index.ndx -o MSD_Z.xvg -lateral z

# --- Load trajectory ---
u = mda.Universe("nvt_auto.tpr", "traj_first1000_unwrapped.xtc")  # positions in Å

# Select atoms of interest (lipids)
ag = u.select_atoms("name NC3 PO4 GL1 GL2 C1A C2A C3A C1B C2B C3B")

# --- Prepare per-lipid COM array ---
residue_group = ag.residues
n_residues = len(residue_group)
n_frames = len(u.trajectory)

coords_z = np.zeros((n_frames, n_residues))  # store Z COM positions in nm



# --- Extract Z coordinates per frame ---
for i, ts in enumerate(u.trajectory):
    # Compute Z coordinate of each residue COM in nm
    z_nm = np.array([res.atoms.center_of_mass()[2] for res in residue_group]) * 0.1  # Å -> nm
    coords_z[i, :] = z_nm

# --- Compute MSD along Z ---
max_lag = n_frames
msd_z = np.zeros(max_lag)

for lag in range(1, max_lag):
    disp2 = 0.0
    count = 0
    for t in range(n_frames - lag):
        dz = coords_z[t + lag, :] - coords_z[t, :]
        dz2 = dz**2
        disp2 += np.sum(dz2)
        count += n_residues
    msd_z[lag] = disp2 / count
    
    
    
# --- Convert from Å² to nm² ---
#msd_z = msd_z * 0.01

# --- Time array ---
dt = 1000  # ps per frame (replace with your actual timestep)
time = np.arange(max_lag) * dt


# --- Fit linear regime for diffusion ---
fit_start = int(0.2 * max_lag)
fit_end = int(0.8 * max_lag)

slope, intercept = np.polyfit(time[fit_start:fit_end], msd_z[fit_start:fit_end], 1)

# Diffusion coefficient along Z (nm^2/ps)
D_z = slope / 2.0       # 1D diffusion
D_z_cm2_s = D_z * 1e-2  # convert nm^2/ps -> cm^2/s

print(f"Slope of MSD fit = {slope:.6f} nm²/ps")
print(f"Estimated diffusion coefficient along Z: D_z = {D_z:.6f} nm²/ps")
print(f"Converted D_z = {D_z_cm2_s:.6e} cm²/s")


# --- Save Z-MSD results to text file ---
with open("msd_z_fit_results.txt", "w") as f:
    f.write(f"Slope of MSD fit along Z (nm^2/ps): {slope:.6f}\n")
    f.write(f"Intercept of MSD fit along Z (nm^2): {intercept:.6f}\n")
    f.write(f"Diffusion coefficient D_z (nm^2/ps): {D_z:.6f}\n")
    f.write(f"Diffusion coefficient D_z (cm^2/s): {D_z_cm2_s:.6e}\n")
    
    

# --- Plot MSD along Z ---
plt.figure(figsize=(7,5))
plt.plot(time, msd_z, 'b-', label="MSD along Z-axis")
plt.plot(time[fit_start:fit_end],
         slope*time[fit_start:fit_end] + intercept,
         'r--', lw=2,
         label=f"Linear fit\nslope={slope:.3f} nm²/ps\nD_z={D_z:.3f} nm²/ps")
plt.xlabel("Lag (ps)")
plt.ylabel("MSD along Z (nm²)")
plt.title("MSD along cylinder axis Z")
plt.legend()
plt.tight_layout()
plt.savefig("MSD_along_Z.png", dpi=300)
plt.show()




# Save MSD data to text file
output_data = np.column_stack((time, msd_z))
np.savetxt("msd_z_Mycode.xvg", output_data, header="lagtime\tMSD_z(nm^2)", fmt="%.6f", delimiter="\t")


