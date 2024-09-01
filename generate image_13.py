import os
import subprocess
import time

# Define the parent folder containing all second-rank folders
parent_folder = "/home/lch/Downloads/3D-RETR-main/ShapeNetVox32/04530566"

# List all folders in the parent folder
second_rank_folders = [folder for folder in os.listdir(parent_folder) if
                       os.path.isdir(os.path.join(parent_folder, folder))]

# Iterate over each second-rank folder
for folder in second_rank_folders:
    # Construct the path to the current second-rank folder
    print(f'\n fodler :{folder}')
    second_rank_folder = os.path.join(parent_folder, folder)

    # List all files in the current second-rank folder
    files = os.listdir(second_rank_folder)
    print(f'\n files:{files}')
    # Filter out OBJ files
    obj_files = [file for file in files if file.endswith('.binvox')]

    # Execute command line operations for each OBJ file in the current folder

    for obj_file in obj_files:
        # Construct the command
        print(f'\n obj file :{obj_file}')
        obj_file_path = os.path.join(second_rank_folder, obj_file)
        print(f'\n base name:{(obj_file_path)}')
        pngf="model.png"
        command = f"blender --background --python  /home/lch/Downloads/bpy-visualization-utils-master/render_binvox_forshapenetdataset.py -- --binvox  {obj_file_path} --output {os.path.join(os.path.splitext(obj_file_path)[0],pngf) }"
        time.sleep(2)
        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Command executed successfully for {obj_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {obj_file}: {e}")
