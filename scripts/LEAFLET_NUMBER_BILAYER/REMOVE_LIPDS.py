import MDAnalysis as mda
import numpy as np

def remove_outer_leaflet_lipids(universe, lipid_resnames=["DLPC"], headgroup_bead="PO4", tailgroup_bead="C3B", n_to_remove=50):
    lipids = universe.select_atoms(f"resname {' '.join(lipid_resnames)}")

    # Select head and tailgroup atoms
    headgroups = lipids.select_atoms(f"name {headgroup_bead}")
    tailgroups = lipids.select_atoms(f"name {tailgroup_bead}")

    coords = headgroups.positions
    coords_tail = tailgroups.positions

    # Compute radial distances in YZ plane
    center_of_mass = lipids.center_of_mass()
    distances = np.linalg.norm(coords[:, 1:3] - center_of_mass[1:3], axis=1)
    distances_tail = np.linalg.norm(coords_tail[:, 1:3] - center_of_mass[1:3], axis=1)
    mean_radius = np.median(distances_tail)

    # Identify outer leaflet by comparing headgroup distances
    outer_mask = distances > mean_radius
    outer_indices = np.where(outer_mask)[0]

    outer_headgroup_atoms = headgroups.atoms[outer_indices]
    outer_resids_all = np.unique(outer_headgroup_atoms.resids)

    if len(outer_resids_all) < n_to_remove:
        raise ValueError(f"Only {len(outer_resids_all)} outer leaflet lipids; cannot remove {n_to_remove}.")

    # Randomly select resids to remove
    resids_to_remove = np.random.choice(outer_resids_all, size=n_to_remove, replace=False)

    # Create an atom group without the selected resids
    atoms_to_keep = universe.select_atoms(f"not (resid {' '.join(map(str, resids_to_remove))})")

    print(f"✅ Removed {n_to_remove} outer leaflet lipids.")
    return atoms_to_keep


### === MAIN SCRIPT === ###

# Load structure
universe = mda.Universe("System_157nm_80micrisecons.gro")
lipid_resnames = ["DLPC"]

# Remove 222 lipids from the outer leaflet
atoms_to_keep = remove_outer_leaflet_lipids(universe, lipid_resnames=lipid_resnames, n_to_remove=200)

# Write new structure
output_gro = "System_new.gro"
atoms_to_keep.write(output_gro)
print(f"\n✅ Modified structure written to: {output_gro}")


# ✅ In plain words
#
# The script:
# - Identifies the outer leaflet lipids by looking at their headgroup positions relative to the bilayer center.
# - Randomly deletes 200 of those lipids.
# - Writes out a new .gro file without the removed lipids.
