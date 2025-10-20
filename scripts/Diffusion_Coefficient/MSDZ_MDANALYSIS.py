import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
import numpy as np
import matplotlib.pyplot as plt

# First : gmx trjconv -s nvt_auto.tpr -f traj_first1000 -o traj_first1000_unwrapped.xtc -pbc nojump
# Match with : echo "0" | gmx msd -f traj_first1000_unwrapped.xtc -s nvt.tpr -n index.ndx -o MSD_Z.xvg -lateral z


# Load test trajectory
u = mda.Universe("nvt_auto.tpr", "traj_first1000_unwrapped.xtc")  # positions in Å


# Compute MSD in Z direction only
msd_calc = msd.EinsteinMSD(u, select='all', msd_type='z', fft=True)
msd_calc.run()

# Convert from Å² to nm²
msd_z = msd_calc.results.timeseries*0.01  
nframes = msd_calc.n_frames

# Define time step between frames (set to your actual trajectory dt)
timestep = 1000 
lagtimes = np.arange(nframes) * timestep


# Plot
fig, ax = plt.subplots()
ax.plot(lagtimes, msd_z, color="black", ls="-", label="MSD in Z-direction")


ax.set_xlabel("Lag time (τ)")
ax.set_ylabel(r"MSD$_z$ (nm$^2$)")
ax.set_title("Mean Square Displacement in Z-direction")
ax.legend()
plt.savefig("msd_z_MD_Analysis.png", dpi=300, bbox_inches="tight")
plt.show()




# Save MSD data to text file
output_data = np.column_stack((lagtimes, msd_z))
np.savetxt("msd_z_MD_Analysis.xvg", output_data, header="lagtime\tMSD_z(nm^2)", fmt="%.6f", delimiter="\t")

