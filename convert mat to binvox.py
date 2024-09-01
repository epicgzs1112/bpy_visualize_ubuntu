import bpy
import numpy as np

# Path to your OBJ file
obj_file = "/home/lch/Downloads/3D-RETR-main/pix3d/model/chair/SS_002/model.obj"

# Path to where you want to save the Binvox file
binvox_file = "/home/lch/Downloads/3D-RETR-main/pix3d/model/chair/SS_002/model.binvox"

# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import OBJ file
result = bpy.ops.import_scene.obj(filepath=obj_file)

# Check if import operation succeeded
if result == {'FINISHED'}:
    # Convert imported object(s) to mesh
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.convert(target='MESH')

    # Select the mesh object
    obj = bpy.context.selected_objects[0]
    mesh = obj.data

    # Determine the bounding box of the mesh
    min_x, max_x = min(v.co.x for v in mesh.vertices), max(v.co.x for v in mesh.vertices)
    min_y, max_y = min(v.co.y for v in mesh.vertices), max(v.co.y for v in mesh.vertices)
    min_z, max_z = min(v.co.z for v in mesh.vertices), max(v.co.z for v in mesh.vertices)

    # Calculate the dimensions of the bounding box
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    # Calculate the resolution of the voxel grid
    resolution = 32  # Adjust as needed
    voxel_size = max(dx, dy, dz) / resolution

    # Create a voxel grid
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Iterate over the vertices and set voxels inside the mesh
    for v in mesh.vertices:
        x = int((v.co.x - min_x) / voxel_size)
        y = int((v.co.y - min_y) / voxel_size)
        z = int((v.co.z - min_z) / voxel_size)
        voxels[x, y, z] = True

    # Write the voxel data to a Binvox file
    with open(binvox_file, 'wb') as f:
        # Write Binvox header
        f.write(b'#binvox 1\n')
        f.write(f'dim {resolution} {resolution} {resolution}\n'.encode())
        f.write(b'translate 0 0 0\n')
        f.write(b'scale 1\n')
        f.write(b'data\n')

        # Write voxel data
        for z in range(resolution):
            for y in range(resolution):
                for x in range(resolution):
                    if voxels[x, y, z]:
                        f.write(b'\x01')  # Write 1 for a voxel present
                    else:
                        f.write(b'\x00')  # Write 0 for a voxel absent

    print("Binvox file saved successfully.")

else:
    print("Failed to import OBJ file.")
