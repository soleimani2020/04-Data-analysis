import numpy as np
import typing
import pandas as pd
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
import time


start_time = time.time()


Max_Mesh_Num = 300#int(sys.argv[1])  
NUMBER_FRAMES = 64#int(sys.argv[2]) 

### find . -type f -name "grid_0.gro"



def find_most_constricted_radius(tpr_file="nvt_auto.tpr", 
                                traj_file="traj_comp.xtc", 
                                segment_width=20,
                                frame=None):
    """
    Find the most constricted radius and its location for each frame in a trajectory.
    """
    
    # Load system
    u = mda.Universe("nvt_auto.tpr", "traj_comp.xtc")
    
    # First pass: Determine global min/max z
    z_min_global = np.inf
    z_max_global = -np.inf

    for ts in u.trajectory:
        ag = u.select_atoms('all')
        z_min_global =  min(z_min_global, np.min(ag.positions[:, 2]))
        z_max_global =  max(z_max_global, np.max(ag.positions[:, 2]))

    cylinder_length_global = z_max_global - z_min_global
    num_segments_global = int(np.ceil(cylinder_length_global / segment_width))
    #print(f"Global z_min: {z_min_global:.2f}, z_max: {z_max_global:.2f}")
    #print(f"Fixed number of segments: {num_segments_global}")

    # Initialize storage
    mean_segment_values = {f'Segment_{i+1}': [] for i in range(num_segments_global)}
    frame_min_radius = []
    frame_min_z = []


    # Iterate over frames
    for ts in u.trajectory[frame:frame+1]:
        #print(f"Processing Frame {ts.frame}")
        ag = u.select_atoms('all')
        #print("ag:\n",ag.positions)
        box = ts.dimensions 
        Lx, Ly, Lz = box[:3]
        z_min_local = np.min(ag.positions[:, 2])
        #print("z_min_local:",z_min_local)
        z_max_local = np.max(ag.positions[:, 2])
        #print("z_max_local:",z_max_local)
        #print("z_max_local:",z_max_local)
        #print("Local cylinder length:",z_max_local-z_min_local)

            
            
        

        frame_radii = []
        frame_zlocs = []

        for segment in range(num_segments_global):
            segment_z_min = z_min_global + segment * segment_width
            segment_z_max = min(segment_z_min + segment_width, z_max_global)
            segment_center_z = 0.5 * (segment_z_min + segment_z_max)

            segment_ag = ag.select_atoms(f"prop z >= {segment_z_min} and prop z < {segment_z_max}")

            if len(segment_ag) == 0:
                mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
                frame_radii.append(np.nan)
                frame_zlocs.append(segment_center_z)
                continue

            max_z_in_segment = np.max(segment_ag.positions[:, 2])
            if (max_z_in_segment - segment_z_min) < (segment_width / 2):
                mean_segment_values[f'Segment_{segment+1}'].append(np.nan)
                frame_radii.append(np.nan)
                frame_zlocs.append(segment_center_z)
                continue

            center_of_mass = segment_ag.center_of_mass()
            distances = np.linalg.norm(segment_ag.positions[:, 0:2] - center_of_mass[0:2], axis=1)
            mean_radius = np.mean(distances)

            mean_segment_values[f'Segment_{segment+1}'].append(mean_radius)
            frame_radii.append(mean_radius)
            frame_zlocs.append(segment_center_z)

        # Find constriction (minimum radius) in this frame
        if np.all(np.isnan(frame_radii)):
            frame_min_radius.append(np.nan)
            frame_min_z.append(np.nan)
        else:
            min_idx = np.nanargmin(frame_radii)
            frame_min_radius.append(frame_radii[min_idx])
            frame_min_z.append(frame_zlocs[min_idx])

    frame_min_z = [z / 10 if not np.isnan(z) else np.nan for z in frame_min_z]
    
    return frame_min_radius, frame_min_z , z_min_local , z_max_local



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
    
    
    def __init__(
        self, 
        membrane_LX: float = 50, 
        membrane_LY: float = 50, 
        membrane_LZ: float =  None, 
        mesh_resolution: int = Max_Mesh_Num,
        most_constricted_radius: float = None,
        location: tuple = None,
        z_min_local: float = None,
        z_max_local: float = None
        ):  
        
        
        self.most_constricted_radius = most_constricted_radius
        self.location = location
        #print("self.location init:",self.location)
        self.z_min_local = z_min_local
        self.z_max_local = z_max_local
        self.membrane_LX=membrane_LX
        self.membrane_LY=membrane_LY
        self.membrane_LZ = z_max_local-z_min_local
        print("self.membrane_LZ init:",self.membrane_LZ)
        self.mesh_resolution = mesh_resolution
        self.membrane_volume = self.membrane_LX * self.membrane_LY * self.membrane_LZ 
        

        
            

    def _get_xyz_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a mesh grid for a given membrane area."""
        mesh_size_X = self.membrane_LX / self.mesh_resolution
        #print(mesh_size_X)
        mesh_size_Y = self.membrane_LY / self.mesh_resolution
        #print(mesh_size_Y)
        mesh_size_Z = self.membrane_LZ / self.mesh_resolution
        #print(mesh_size_Z)

        grid_volume = mesh_size_X * mesh_size_Y * mesh_size_Z
        
        #print("mesh_size_X:", mesh_size_X)
        #print("mesh_size_Y:", mesh_size_Y)
        #print("mesh_size_Z:", mesh_size_Z)
        
        
        
    
        # Generate VTK header
        vtk_header = f"""# vtk DataFile Version 2.0
        Random MD density example
        ASCII
        DATASET STRUCTURED_POINTS
        DIMENSIONS {self.mesh_resolution} {self.mesh_resolution} {self.mesh_resolution}
        ORIGIN 0.0 0.0 0.0
        SPACING {mesh_size_X} {mesh_size_Y} {mesh_size_Z}
        POINT_DATA {self.mesh_resolution*self.mesh_resolution*self.mesh_resolution}
        SCALARS density float
        LOOKUP_TABLE default
        """

        # Example: generate dummy density values (replace with your computed densities)
        density_values = [0.0 for _ in range(self.mesh_resolution*self.mesh_resolution*self.mesh_resolution)]

        # Write VTK file
        output_file = "density.vtk"
        with open(output_file, "w") as f:
            f.write(vtk_header)


        

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
        #print("z_mesh:",z_mesh)
        #print("Mesh_NUMBER:", Mesh_NUMBER)
        

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


    all_frames_data = []  # List to store data from all frames
    
    @classmethod     
    

    def process_mesh(cls, x_mesh, y_mesh, z_mesh, mesh_size_X, mesh_size_Y, 
                    mesh_size_Z, Mesh_NUMBER, mesh_resolution, z_constriction, xyz_i, 
                    frame, max_z_threshold=1000, min_z_threshold=-1000):        
        """Process a single frame in 3D"""
        try:
            selected_atoms_info = {}
            empty_grids = 0
            

            # Subtracting z_constriction moves the entire membrane so that the constriction is at z = 0.
            xyz_i[:, 6] = xyz_i[:, 6].astype(float) - z_constriction

            # Calculates the minimum and maximum z-values in the system after shifting.
            min_z = np.min(xyz_i[:, 6].astype(float))
            max_z = np.max(xyz_i[:, 6].astype(float))
            pbc_z = max_z - min_z
            
            # Creates a new z-mesh from min_z to max_z in steps of mesh_size_Z; array of all starting points of slices.
            z_mesh_updated = np.arange(min_z, max_z, mesh_size_Z)
            num_z_cells = len(z_mesh_updated)
            
            # Get actual minimum coordinates for proper grid alignment
            x_coords = xyz_i[:, 4].astype(float)
            y_coords = xyz_i[:, 5].astype(float)
            z_coords = xyz_i[:, 6].astype(float)
            atom_names = xyz_i[:, 2]
            
            x_min_actual = 0#np.min(x_coords)-50  # Smallest x-coordinate in system
            y_min_actual = 0#np.min(y_coords)-50  # Smallest y-coordinate in system  
            z_min_actual = np.min(z_coords)  # Smallest z-coordinate in system
            
            # Precompute grid indices for all particles relative to actual minima
            x_indices = ((x_coords - x_min_actual) // mesh_size_X).astype(int)
            y_indices = ((y_coords - y_min_actual) // mesh_size_Y).astype(int)
            z_indices = ((z_coords - z_min_actual) // mesh_size_Z).astype(int)
            
            # Filter for valid indices and NC3 atoms
            valid_mask = (
                (x_indices >= 0) & (x_indices < mesh_resolution) &
                (y_indices >= 0) & (y_indices < mesh_resolution) &
                (z_indices >= 0) & (z_indices < num_z_cells) &
                (atom_names == 'NC3')
            )
            
            valid_indices = np.where(valid_mask)[0]
            
            # Group particles by grid cell
            grid_particles = {}
            for idx in valid_indices:
                grid_key = (x_indices[idx], y_indices[idx], z_indices[idx])
                if grid_key not in grid_particles:
                    grid_particles[grid_key] = []
                grid_particles[grid_key].append(idx)
            
            # Process each grid cell and collect atoms (including subsequent atoms)
            for grid_key, particle_indices in grid_particles.items():
                selected_atoms_info[grid_key] = []
                
                for idx in particle_indices:
                    # Add the NC3 atom and next 9 atoms
                    end_idx = min(idx + 10, len(xyz_i))
                    for m in range(idx, end_idx):
                        selected_atoms_info[grid_key].append(xyz_i[m])
            
            # Count empty grids
            total_grids = mesh_resolution * mesh_resolution * num_z_cells
            print("mesh_resolution:",mesh_resolution)
            print("num_z_cells:",num_z_cells)
            empty_grids = total_grids - len(grid_particles)
            
            Tota_Grid_Number = num_z_cells * mesh_resolution * mesh_resolution
            print("Tota_Grid_Number:", Tota_Grid_Number)



            # Create a summary DataFrame to store grid information for ALL grids

            grid_summary_data = []
                        
            for i in range(mesh_resolution):
                for j in range(mesh_resolution):
                    for k in range(num_z_cells):
                        grid_key = (i, j, k)
                        folder_index = (i + j * mesh_resolution + k * mesh_resolution * mesh_resolution)
                        
                        if grid_key in selected_atoms_info and selected_atoms_info[grid_key]:
                            atoms_info_list = selected_atoms_info[grid_key]
                            data_array = np.array(atoms_info_list)
                            num_atoms = len(data_array)
                        else:
                            num_atoms = 0
                        
                        # Append data to the list (not creating DataFrame yet)
                        grid_summary_data.append({
                            'grid_key': grid_key,
                            'num_atoms': num_atoms,
                            'frame': frame
                        })
            
            # Add this frame's data to the global list
            grid_summary_df = pd.DataFrame(grid_summary_data)
            output_file = "grid_summary_all.csv"
            grid_summary_df.to_csv(
                output_file, 
                mode='a', 
                header=not os.path.exists(output_file), 
                index=False
            )
            
            return 0

        except Exception as e:
            print(f"Error processing 3D mesh: {str(e)}")
            return -1

    
    
def process_frame(frame):
    folder_path = '/p/scratch/smt/ALIREZA/BACKUP_30MICROSECONDS/Density_profile/CONF'
    # Find constriction
    most_constricted_radius, location ,  z_min_local , z_max_local = find_most_constricted_radius(frame=frame)    
    read_gro_instance = ReadGro(fname=os.path.join(folder_path, f"conf{frame}.gro"))
    gro_data = read_gro_instance.gro_data
    xyz_i = gro_data[['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']].values
    z_min_local_nm = np.min(xyz_i[:, 6].astype(float))
    z_max_local_nm = np.max(xyz_i[:, 6].astype(float))
    mesh_generator = APL_ANALYSIS(
    most_constricted_radius=most_constricted_radius,
    location=location[0],
    z_min_local=z_min_local_nm,
    z_max_local=z_max_local_nm
    )  
    
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
        z_constriction=location[0],
        xyz_i=xyz_i,
        frame=frame
    )
    
    #print(f"Completed 3D analysis of frame {frame}")
    return result    
    
    

if __name__ == "__main__":
    frames = range(0, NUMBER_FRAMES)
    num_processes = multiprocessing.cpu_count()  
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_frame, frames)
        
            



# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.6f} seconds")
print(f"Total execution time: {elapsed_time/60:.6f} minutes")




# import pandas as pd
# import numpy as np
# 
# mesh_resolution = num_z_cells = 3
# 
# 
# # Read the CSV file
# df = pd.read_csv("grid_summary_all.csv")
# 
# # Convert grid_key from string to tuple if it's stored as string
# if isinstance(df['grid_key'].iloc[0], str):
#     df['grid_key'] = df['grid_key'].apply(lambda x: eval(x) if isinstance(x, str) else x)
# 
# # Create a dictionary to store num_atoms for each grid cell
# cell_atoms = {}
# 
# # Group by grid_key and collect all num_atoms values
# for grid_key, group in df.groupby('grid_key'):
#     num_atoms_list = group['num_atoms'].tolist()
#     cell_atoms[grid_key] = num_atoms_list
# 
# # Calculate averages for each grid cell
# cell_averages = {}
# for grid_key, atoms_list in cell_atoms.items():
#     average_atoms = np.mean(atoms_list)
#     cell_averages[grid_key] = average_atoms
#     print(f"Grid cell {grid_key}: Average atoms = {average_atoms:.2f}")
# 
# # If you want to create a 3D array of averages
# average_matrix = np.zeros((mesh_resolution, mesh_resolution, num_z_cells))
# 
# for (i, j, k), avg in cell_averages.items():
#     average_matrix[i, j, k] = avg
# 
# print(f"Overall average across all cells: {np.mean(list(cell_averages.values())):.2f}")
# 
# 
# 
# 
# 
# 
# 
# 





print("End of the code . good luck")




# import pandas as pd
# import numpy as np
# 
# 
# # Read the CSV file
# df = pd.read_csv("grid_summary_all.csv")
# 
# # Convert grid_key from string to tuple if it's stored as string
# if isinstance(df['grid_key'].iloc[0], str):
#     df['grid_key'] = df['grid_key'].apply(lambda x: eval(x) if isinstance(x, str) else x)
# 
# # Create a dictionary to store num_atoms for each grid cell
# cell_atoms = {}
# 
# # Group by grid_key and collect all num_atoms values
# for grid_key, group in df.groupby('grid_key'):
#     num_atoms_list = group['num_atoms'].tolist()
#     cell_atoms[grid_key] = num_atoms_list
# 
# # Calculate averages for each grid cell
# cell_averages = {}
# for grid_key, atoms_list in cell_atoms.items():
#     average_atoms = np.mean(atoms_list)
#     cell_averages[grid_key] = average_atoms
#     print(f"Grid cell {grid_key}: Average atoms = {average_atoms:.2f}")
# 
# # If you want to create a 3D array of averages
# average_matrix = np.zeros((mesh_resolution, mesh_resolution, num_z_cells))
# 
# for (i, j, k), avg in cell_averages.items():
#     average_matrix[i, j, k] = avg
# 
# print(f"Overall average across all cells: {np.mean(list(cell_averages.values())):.2f}")
# 
# 
# 
# 
# 
# # Read the CSV file
# df = pd.read_csv("grid_summary_all.csv")
# 
# # Convert grid_key to tuple if needed
# if isinstance(df['grid_key'].iloc[0], str):
#     df['grid_key'] = df['grid_key'].apply(lambda x: eval(x))
# 
# # Calculate average for each grid cell
# cell_averages = df.groupby('grid_key')['num_atoms'].mean().to_dict()
# 
# # Write averages to file in the specified order
# with open("normalized_values.txt", "w") as f:
#     for k in range(num_z_cells):          # z loop
#         for j in range(mesh_resolution):  # y loop
#             for i in range(mesh_resolution):  # x loop
#                 grid_key = (i, j, k)
#                 
#                 # Get the average value for this grid cell
#                 avg_value = cell_averages.get(grid_key, 0.0)
#                 
#                 # Write to file
#                 f.write(f"{avg_value}\n")
# 
# print(f"Total values written: {mesh_resolution * mesh_resolution * num_z_cells}")
# 
