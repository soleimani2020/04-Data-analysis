import numpy as np
import sys
import typing
import pandas as pd
import pandas as pd
import numpy as np
import sys
import typing
import os
import subprocess
from subprocess import call
import shutil
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import islice
import MDAnalysis as mda


### find . -type f -name "grid_0.gro"


NUMBER_FRAMES = 1 

# # Load system
# u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")
# 
# 
# segment_width = 20  # Angstroms
# 
# # First pass: Determine global min/max z
# z_min_global = np.inf
# z_max_global = -np.inf
# 
# for ts in u.trajectory:
#     ag = u.select_atoms('name C3B')
#     z_min_global = min(z_min_global, np.min(ag.positions[:, 2]))
#     z_max_global = max(z_max_global, np.max(ag.positions[:, 2]))
# 
# cylinder_length_global = z_max_global - z_min_global
# num_segments_global = int(np.ceil(cylinder_length_global / segment_width))
# print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
# print(f"Fixed number of segments: {num_segments_global}")
# 
# # Initialize storage
# mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}
# frame_min_radius = []
# frame_min_z = []
# 
# # Iterate over frames
# for ts in u.trajectory[:10]:
#     print(f"Processing Frame {ts.frame}")
#     ag = u.select_atoms('name C3B')
# 
#     frame_radii = []
#     frame_zlocs = []
# 
#     for segment in range(num_segments_global):
#         segment_z_min = z_min_global + segment * segment_width
#         segment_z_max = min(segment_z_min + segment_width, z_max_global)
#         segment_center_z = 0.5 * (segment_z_min + segment_z_max)
# 
#         segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")
# 
#         if len(segment_ag) == 0:
#             mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
#             frame_radii.append(np.nan)
#             frame_zlocs.append(segment_center_z)
#             continue
# 
#         max_z_in_segment = np.max(segment_ag.positions[:, 2])
#         if (max_z_in_segment - segment_z_min) < (segment_width / 2):
#             mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
#             frame_radii.append(np.nan)
#             frame_zlocs.append(segment_center_z)
#             continue
# 
#         center_of_mass = segment_ag.center_of_mass()
#         distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1)
#         mean_radius = np.mean(distances)
# 
#         mean_segment_values[f'Segment_{segment+1}'].append(mean_radius)
#         frame_radii.append(mean_radius)
#         frame_zlocs.append(segment_center_z)
# 
#     # Find constriction (minimum radius) in this frame
#     if np.all(np.isnan(frame_radii)):
#         frame_min_radius.append(np.nan)
#         frame_min_z.append(np.nan)
#     else:
#         min_idx = np.nanargmin(frame_radii)
#         frame_min_radius.append(frame_radii[min_idx])
#         frame_min_z.append(frame_zlocs[min_idx])
# 
# # Convert to DataFrames
# mean_segment_df = pd.DataFrame(mean_segment_values)
# min_radius_df = pd.DataFrame({
#     "Frame": range(len(frame_min_radius)),
#     "Min_Radius": frame_min_radius,
#     "Min_Radius_z": frame_min_z
# })
# 
# print(min_radius_df.head())



#############
#############
#############
 


class ReadGro:
    """reading GRO file based on the doc"""

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    line_len: int = 45  # Length of the lines in the data file
    gro_data: pd.DataFrame  # All the informations in the file
    # The follwings will set in __process_header_tail method:
    title: str  # Name of the system
    number_atoms: int  # Total number of atoms in the system
    pbc_box: str  # Size of the box (its 3 floats but save as a string)

    def __init__(self,
                 fname: str,  # Name of the input file
                 ) -> None:
        self.gro_data = self.read_gro(fname)

    def read_gro(self,
                 fname: str  # gro file name
                 ) -> pd.DataFrame:
        """read gro file lien by line"""
        counter: int = 0  # To count number of lines
        processed_line: list[dict[str, typing.Any]] = []  # All proccesed lines
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                line = f_r.readline()
                if len(line) != self.line_len:
                    self.__process_header_tail(line.strip(), counter)
                else:
                    processed_line.append(self.__process_line(line.rstrip()))
                counter += 1
                if not line.strip():
                    break
        ReadGro.info_msg += f'\tFile name is {fname}\n'        
        ReadGro.info_msg += f'\tSystem title is {self.title}\n'
        ReadGro.info_msg += f'\tNumber of atoms is {self.number_atoms}\n'
        ReadGro.info_msg += f'\tBox boundary is {self.pbc_box}\n'
        return pd.DataFrame(processed_line)

    @staticmethod
    def __process_line(line: str  # Data line
                       ) -> dict[str, typing.Any]:
        """process lines of information"""
        resnr = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        a_x = float(line[20:28])
        a_y = float(line[28:36])
        a_z = float(line[36:44])
        processed_line: dict[str, typing.Any] = {
                                                 'residue_number': resnr,
                                                 'residue_name': resname,
                                                 'atom_name': atomname,
                                                 'atom_id': atomnr,
                                                 'x': a_x,
                                                 'y': a_y,
                                                 'z': a_z,
                                                }
        return processed_line

    def __process_header_tail(self,
                              line: str,  # Line in header or tail
                              counter: int  # Line number
                              ) -> None:
        """Get the header, number of atoms, and box size"""
        if counter == 0:
            self.title = line
        elif counter == 1:
            self.number_atoms = int(line)
        elif counter == self.number_atoms + 2:
            self.pbc_box = line




class APL_ANALYSIS:
    
    
    def __init__(self, membrane_LX: float = 50, membrane_LY: float = 50, membrane_LZ: float =  74.55861, mesh_resolution: int = 3):  
        self.membrane_LX=membrane_LX
        self.membrane_LY=membrane_LY
        self.membrane_LZ = membrane_LZ  
        self.mesh_resolution = mesh_resolution
        self.membrane_volume = self.membrane_LX * self.membrane_LY * self.membrane_LZ  
        
            

    def _get_xyz_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a mesh grid for a given membrane area."""
        mesh_size_X = self.membrane_LX / self.mesh_resolution
        print(mesh_size_X)
        mesh_size_Y = self.membrane_LY / self.mesh_resolution
        print(mesh_size_Y)
        mesh_size_Z = self.membrane_LZ / self.mesh_resolution
        print(mesh_size_Z)

        grid_volume = mesh_size_X * mesh_size_Y * mesh_size_Z
        
        print("mesh_size_X:", mesh_size_X)
        print("mesh_size_Y:", mesh_size_Y)
        print("mesh_size_Z:", mesh_size_Z)
        print("grid_volume:", grid_volume)
        

        # Create 3D meshgrid
        x_mesh, y_mesh, z_mesh = np.meshgrid(
            np.arange(0.0, self.membrane_LX, mesh_size_X),
            np.arange(0.0, self.membrane_LY, mesh_size_Y),
            np.arange(0.0, self.membrane_LZ, mesh_size_Z)
        )
        

        L1 = len(x_mesh)
        L2 = len(y_mesh)
        L3 = len(z_mesh)
        Mesh_NUMBER = L1 * L2 * L3
        print("Mesh_NUMBER:", Mesh_NUMBER)
        

        return x_mesh, y_mesh, z_mesh, grid_volume, Mesh_NUMBER,   self.mesh_resolution,  mesh_size_X, mesh_size_Y, mesh_size_Z,  self.membrane_volume  
    
    

    @staticmethod
    def write_gromacs_gro(gro_data: pd.DataFrame,
                          output_directory: str,
                          filename: str,  # Name of the output file
                          pbc_box=None,
                          title=None
                          ) -> None:
        """Write DataFrame to a GROMACS gro file."""
        
        
        df_i: pd.DataFrame = gro_data.copy()
        
        output_file_path = os.path.join(output_directory,filename)
        
        with open(output_file_path, 'w', encoding='utf8') as gro_file:
            if title:
                gro_file.write(f'{title}')  # Add a comment line
            gro_file.write(f'{len(df_i)}\n')  # Write the total number of atoms
            for _, row in df_i.iterrows():
                line = f'{row["residue_number"]:>5}' \
                       f'{row["residue_name"]:<5}' \
                       f'{row["atom_name"]:>5}' \
                       f'{row["atom_id"]:>5}' \
                       f'{row["x"]:8.3f}' \
                       f'{row["y"]:8.3f}' \
                       f'{row["z"]:8.3f}\n'
                gro_file.write(line)
            if pbc_box:
                gro_file.write(f'{pbc_box}\n')


    
    
    
    @classmethod        
    def process_mesh(cls, x_mesh, y_mesh, z_mesh, mesh_size_X, mesh_size_Y, 
                    mesh_size_Z, Mesh_NUMBER, mesh_resolution, xyz_i, 
                    frame, max_z_threshold=1000, min_z_threshold=-1000):        
        """Process a single frame in 3D"""
        try:
            selected_atoms_info = {}
            empty_grids = 0

            for i in range(x_mesh.shape[0]):
                for j in range(x_mesh.shape[1]):
                    for k in range(z_mesh.shape[2]):
                        x_min = x_mesh[i, j, k]
                        x_max = x_min +  mesh_size_X
                        y_min = y_mesh[i, j, k]
                        y_max = y_min + mesh_size_Y
                        z_min = z_mesh[i, j, k]
                        z_max = z_min + mesh_size_Z

                        # Mask for 'NC3' atoms in current 3D cell
                        mask_nc3 = (xyz_i[:, 2] == 'NC3')
                        ind_in_mesh_nc3 = np.where(
                            (xyz_i[:, 4] >= x_min) & (xyz_i[:, 4] < x_max) &
                            (xyz_i[:, 5] >= y_min) & (xyz_i[:, 5] < y_max) &
                            (xyz_i[:, 6] >= z_min) & (xyz_i[:, 6] < z_max) &
                            mask_nc3
                        )[0]

                        grid_key = (i, j, k)
                        selected_atoms_info[grid_key] = []

                        for idx in ind_in_mesh_nc3:
                            selected_atoms_info[grid_key].append(xyz_i[idx])
                            # Add next 9 atoms
                            for m in range(idx + 1, min(idx + 10, len(xyz_i))):
                                selected_atoms_info[grid_key].append(xyz_i[m])

                        if not ind_in_mesh_nc3.size:
                            empty_grids += 1

            print(f"Frame {frame}: {empty_grids} empty grids out of {Mesh_NUMBER}")

            # Create output folders
            for folder_index in range(Mesh_NUMBER):
                os.makedirs(f"Eolder_{folder_index}", exist_ok=True)

            # Process each grid cell
            for grid_key, atoms_info_list in selected_atoms_info.items():
                folder_index = (grid_key[0] + 
                               grid_key[1] * mesh_resolution + 
                               grid_key[2] * mesh_resolution * mesh_resolution)
                folder_path = f"Eolder_{folder_index}"
                
                if atoms_info_list:  # Only write files for non-empty grids
                    data_array = np.array(atoms_info_list)
                    df = pd.DataFrame(data_array, columns=[
                        'residue_number', 'residue_name', 'atom_name', 
                        'atom_id', 'x', 'y', 'z'])
                    
                    APL_ANALYSIS.write_gromacs_gro(
                        df, folder_path, f"grid_{frame}.gro",
                        pbc_box="50.00000  50.00000  74.55861",
                        title=f"Atoms in grid {grid_key}, frame {frame}\n"
                    )
                else:
                    # Create empty marker file
                    with open(os.path.join(folder_path, f"EMPTY_{frame}.txt"), 'w') as f:
                        f.write(f"Grid {grid_key} was empty in frame {frame}")

            return 0
            
        except Exception as e:
            print(f"Error processing 3D mesh: {str(e)}")
            return -1
    
    


    
    
def process_frame(frame):
    read_gro_instance = ReadGro(fname="conf"+str(frame)+".gro")
    gro_data = read_gro_instance.gro_data
    xyz_i = gro_data[['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']].values
    mesh_generator = APL_ANALYSIS()  
    # Generate 3D mesh grid
    (x_mesh, y_mesh, z_mesh, grid_volume, Mesh_NUMBER, 
     mesh_resolution, mesh_size_X, mesh_size_Y, mesh_size_Z, 
     membrane_volume) = mesh_generator._get_xyz_grid()
    
    # Process the frame with 3D mesh
    result = mesh_generator.process_mesh(
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        z_mesh=z_mesh,
        mesh_size_X=mesh_size_X,
        mesh_size_Y=mesh_size_Y,
        mesh_size_Z=mesh_size_Z,
        Mesh_NUMBER=Mesh_NUMBER,
        mesh_resolution=mesh_resolution,
        xyz_i=xyz_i,
        frame=frame
    )
    
    print(f"Completed 3D analysis of frame {frame}")
    return result    
    
    

if __name__ == "__main__":
    frames = range(0, 3)
    num_processes = multiprocessing.cpu_count()  
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_frame, frames)
        
            




print("End of the code . good luck")
