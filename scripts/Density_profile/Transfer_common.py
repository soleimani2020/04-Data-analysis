import os
import shutil

# Get a list of all folders in the directory
folders = [name for name in os.listdir() if os.path.isdir(name)]
print(folders)

# Specify the files to be copied
#files_to_copy = ['topol.top', 'pull.mdp', 'molecule.itp', 'martini_v3.0.0_solvents_v1.itp', 'martini_v3.0.0_phospholipids_v1.itp', 'martini_v3.0.0.itp' , 'index.ndx' , 'go_nbparams.itp', 'go_atomtypes.itp']


files_to_copy = ['dens.py']

# Get the full path of the current directory
current_directory = os.getcwd()

# Copy the files to each folder
for folder in folders:
    for file_to_copy in files_to_copy:
        source_file = os.path.join(current_directory, file_to_copy)
        if not os.path.exists(source_file):
            print(f"Source file '{source_file}' not found.")
            continue
        
        destination_folder = os.path.join(current_directory, folder, file_to_copy)
        try:
            shutil.copy(source_file, destination_folder)
            print(f"File '{file_to_copy}' copied to '{destination_folder}'.")
        except Exception as e:
            print(f"Error copying file '{file_to_copy}' to '{destination_folder}': {e}")
