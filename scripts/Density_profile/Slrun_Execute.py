import os
import subprocess
import sys



num_files = int(sys.argv[2])

base_folder = '/p/scratch/memm/ALIREZA/AAA_PAPER_RESULTS/Cylinder/DRY_MARTINI/DR_MARTINI_EQUILIBRIATION/RAHMAN_STEP3/TC_4/BACKUP_30MICROSECONDS/Density_profile/'

# List all Eolder_* folders in base_folder
folders = [f for f in os.listdir(base_folder) if f.startswith("Eolder_") and os.path.isdir(os.path.join(base_folder, f))]
#print(f"Found {len(folders)} folders: {folders}")

for folder in folders:
    conf_folder = os.path.join(base_folder, folder)
    dens_file = os.path.join(conf_folder, "dens.py")    

    if os.path.exists(dens_file):
        # Save current directory to return later
        orig_dir = os.getcwd()
        os.chdir(conf_folder)  # Change to folder containing dens.py

        # Run dens.py with num_files as argument
        subprocess.run(["python3", "dens.py", str(num_files)], check=True)
        #print(f"Executed")

        # Go back to original directory
        os.chdir(orig_dir)
    else:
        print(f"No 'dens.py' found in {conf_folder}, skipping.")
