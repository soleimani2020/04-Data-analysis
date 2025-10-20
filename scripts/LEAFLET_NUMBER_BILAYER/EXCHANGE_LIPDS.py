import MDAnalysis as mda
import numpy as np
import random

def count_lipids_and_transfer(universe, lipid_resnames=["DLPC"], headgroup_bead="PO4", tailgroup_bead="C3B", n_to_transfer=50):
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

    # Classify leaflet
    outer_mask = distances > mean_radius
    inner_indices = np.where(~outer_mask)[0]

    inner_headgroup_atoms = headgroups.atoms[inner_indices]
    inner_resids_all = np.unique(inner_headgroup_atoms.resids)

    # Filter inner lipids by COM X range: 25 < x < 40 nm (250 < x < 400 Å)
    inner_resids_filtered = []
    for resid in inner_resids_all:
        lipid = universe.select_atoms(f"resid {resid}")
        com_x = lipid.center_of_mass()[0]
        if 250.0 < com_x < 600.0:
            inner_resids_filtered.append(resid)

    if len(inner_resids_filtered) < n_to_transfer:
        raise ValueError(f"Only {len(inner_resids_filtered)} inner lipids in 25 < x < 40 nm range; cannot move {n_to_transfer}.")

    selected_resids = np.random.choice(inner_resids_filtered, size=n_to_transfer, replace=False)
    delta_r = (np.mean(distances[outer_mask]) - np.mean(distances[inner_indices])) -5

    coords = universe.atoms.positions.copy()

    # Define bead swap pairs for mirroring
    bead_pairs = [
        ("NC3", "C3B"),
        ("PO4", "C2B"),
        ("GL1", "C1B"),
        ("GL2", "C3A"),
        ("C1A", "C2A")
    ]

    for resid in selected_resids:
        lipid = universe.select_atoms(f"resid {resid}")

        # Step 1: Shift in YZ plane
        for atom in lipid:
            vec_yz = coords[atom.index, 1:3] - center_of_mass[1:3]
            norm = np.linalg.norm(vec_yz)
            if norm == 0:
                continue
            unit_vec = vec_yz / norm
            coords[atom.index, 1:3] += delta_r * unit_vec

        # Step 2: Mirror selected beads
        for bead1, bead2 in bead_pairs:
            try:
                atom1 = lipid.select_atoms(f"name {bead1}")[0]
                atom2 = lipid.select_atoms(f"name {bead2}")[0]
                coords[atom1.index], coords[atom2.index] = coords[atom2.index].copy(), coords[atom1.index].copy()
            except IndexError:
                print(f"⚠️ Missing bead in resid {resid}: {bead1} or {bead2}")

    # Assign new coordinates
    universe.atoms.positions = coords
    print(f"✅ Transferred and mirrored {n_to_transfer} lipids from inner to outer leaflet in 25 < x < 40 nm.")
    return selected_resids


### === MAIN SCRIPT === ###

# Load GRO file only
universe = mda.Universe("Bonf.gro")
lipid_resnames = ["DLPC"]

# Transfer lipids with X-range filtering
selected_resids = count_lipids_and_transfer(universe, lipid_resnames=lipid_resnames, n_to_transfer=50)



# Write new structure
output_gro = "modified_structure.gro"
universe.atoms.write(output_gro)
print(f"\n✅ Modified structure written to: {output_gro}")
