import os
import argparse
import sys
import os
import numpy as np

class Voxels(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """

    def __init__(self, data, dims, translate, scale):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale)

    def write(self, fp):
        write(self, fp)

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp):
    """ Read binary binvox format as array.

    Returns the model with accompanying metadata.

    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)

    return Voxels(data, dims, translate, scale)

def read_as_coord_array(fp, fix_coords=True):
    """ Read binary binvox format as coordinates.

    Returns binvox model with voxels in a "coordinate" representation, i.e.  an
    3 x N array where N is the number of nonzero voxels. Each column
    corresponds to a nonzero voxel and the 3 rows are the (x, z, y) coordinates
    of the voxel.  (The odd ordering is due to the way binvox format lays out
    data).  Note that coordinates refer to the binvox voxels, without any
    scaling or translation.

    Use this to save memory if your model is very sparse (mostly empty).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

    values, counts = raw_data[::2], raw_data[1::2]

    sz = np.prod(dims)
    index, end_index = 0, 0
    end_indices = np.cumsum(counts)
    indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

    values = values.astype(np.bool)
    indices = indices[values]
    end_indices = end_indices[values]

    nz_voxels = []
    for index, end_index in zip(indices, end_indices):
        nz_voxels.extend(range(index, end_index))
    nz_voxels = np.array(nz_voxels)
    # TODO are these dims correct?
    # according to docs,
    # index = x * wxh + z * width + y; // wxh = width * height = d * d

    x = nz_voxels / (dims[0]*dims[1])
    zwpy = nz_voxels % (dims[0]*dims[1]) # z*w + y
    z = zwpy / dims[0]
    y = zwpy % dims[0]

    data = np.vstack((x, y, z))

    #return Voxels(data, dims, translate, scale)
    return Voxels(np.ascontiguousarray(data), dims, translate, scale)

def dense_to_sparse(voxel_data, dtype=np.int):
    """ From dense representation to sparse (coordinate) representation.
    No coordinate reordering.
    """
    if voxel_data.ndim!=3:
        raise ValueError('voxel_data is wrong shape; should be 3D array.')
    return np.asarray(np.nonzero(voxel_data), dtype)

def sparse_to_dense(voxel_data, dims, dtype=np.bool):
    if voxel_data.ndim!=2 or voxel_data.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = voxel_data.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out

#def get_linear_index(x, y, z, dims):
    #""" Assuming xzy order. (y increasing fastest.
    #TODO ensure this is right when dims are not all same
    #"""
    #return x*(dims[1]*dims[2]) + z*dims[1] + y

def write(voxel_model, fp):
    """ Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense format.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    fp.write('#binvox 1\n')
    fp.write('dim '+' '.join(map(str, voxel_model.dims))+'\n')
    fp.write('translate '+' '.join(map(str, voxel_model.translate))+'\n')
    fp.write('scale '+str(voxel_model.scale)+'\n')
    fp.write('data\n')

    voxels_flat = dense_voxel_data.flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr==255:
                fp.write(chr(state))
                fp.write(chr(ctr))
                ctr = 0
        else:
            # if switch state, dump
            fp.write(chr(state))
            fp.write(chr(ctr))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(chr(state))
        fp.write(chr(ctr))

class LogLevel:
    """
    Defines color of log message.
    """

    INFO = '\033[94m'
    """ (string) Blue. """
    WARNING = '\033[93m'
    """ (string) Yellow. """
    ERROR = '\033[91m\033[1m'
    """ (string) Red. """
    ENDC = '\033[0m'
    """ (string) End of color. """


def log(output,level=LogLevel.INFO):
    """
    Log message.

    :param output: message
    :param level: LogLevel
    """

    sys.stderr.write(level)
    sys.stderr.write(str(output))
    sys.stderr.write(LogLevel.ENDC)
    sys.stderr.write("\n")
    sys.stderr.flush()


# This makes sure that Blender's NumPy is loaded first.
# The path needs to be adapted before usage.
# Example:
#   blender_package_path = '~/blender-2.79-linux-glibc219-x86_64/2.79/python/lib/python3.5/site-packages/'
if not 'BLENDER_PACKAGE_PATH' in globals():
    BLENDER_PACKAGE_PATH = None
    BLENDER_PACKAGE_PATH = '/snap/blender/20/2.79/python/lib/python3.5/site-packages'
if BLENDER_PACKAGE_PATH is None:
    log('Open blender_utils.py and set BLENDER_PACKAGE_PATH before usage, check documentation!', LogLevel.ERROR)
    exit()
if not os.path.exists(BLENDER_PACKAGE_PATH):
    log('The set BLENDER_PACKAGE_PATH does not exist, check documentation!', LogLevel.ERROR)
    exit()

sys.path.insert(1, os.path.realpath(BLENDER_PACKAGE_PATH))
import bpy
import bmesh
import math
import numpy as np
sphere_base_mesh = None
cube_base_mesh = None
from bpy import *
#####
#
# Copyright 2014 Alex Tsui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#####

#
# http://wiki.blender.org/index.php/Dev:2.5/Py/Scripts/Guidelines/Addons
#
import os
import bpy
import mathutils
from bpy.props import (BoolProperty,
    FloatProperty,
    StringProperty,
    EnumProperty,
    )
from bpy_extras.io_utils import (ImportHelper,
    ExportHelper,
    unpack_list,
    unpack_face_list,
    axis_conversion,
    )

#if "bpy" in locals():
#    import imp
#    if "import_off" in

bl_info = {
    "name": "OFF format",
    "description": "Import-Export OFF, Import/export simple OFF mesh.",
    "author": "Alex Tsui, Mateusz Kłoczko",
    "version": (0, 3),
    "blender": (2, 74, 0),
    "location": "File > Import-Export",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.5/Py/"
                "Scripts/My_Script",
    "category": "Import-Export"}

class ImportOFF(bpy.types.Operator, ImportHelper):
    """Load an OFF Mesh file"""
    bl_idname = "import_mesh.off"
    bl_label = "Import OFF Mesh"
    filename_ext = ".off"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='Y',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Z',
            )

    def execute(self, context):
        #from . import import_off

        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
        ))
        global_matrix = axis_conversion(from_forward=self.axis_forward,
            from_up=self.axis_up,
            ).to_4x4()

        mesh = load(self, context, **keywords)
        if not mesh:
            return {'CANCELLED'}

        scene = bpy.context.scene
        obj = bpy.data.objects.new(mesh.name, mesh)
        scene.objects.link(obj)
        scene.objects.active = obj
        obj.select = True

        obj.matrix_world = global_matrix

        scene.update()

        return {'FINISHED'}

class ExportOFF(bpy.types.Operator, ExportHelper):
    """Save an OFF Mesh file"""
    bl_idname = "export_mesh.off"
    bl_label = "Export OFF Mesh"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )
    check_extension = True
    filename_ext = ".off"

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='Y',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Z',
            )
    use_colors = BoolProperty(
            name="Vertex Colors",
            description="Export the active vertex color layer",
            default=False,
            )

    def execute(self, context):
        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
            'check_existing',
        ))
        global_matrix = axis_conversion(to_forward=self.axis_forward,
            to_up=self.axis_up,
            ).to_4x4()
        keywords['global_matrix'] = global_matrix
        return save(self, context, **keywords)

def menu_func_import(self, context):
    self.layout.operator(ImportOFF.bl_idname, text="OFF Mesh (.off)")

def menu_func_export(self, context):
    self.layout.operator(ExportOFF.bl_idname, text="OFF Mesh (.off)")

def register():
    bpy.utils.register_module(__name__)

    bpy.types.INFO_MT_file_import.append(menu_func_import)
    bpy.types.INFO_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)

def load(operator, context, filepath):
    # Parse mesh from OFF file
    # TODO: Add support for NOFF and COFF
    filepath = os.fsencode(filepath)
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    use_colors = (first_line == 'COFF')
    colors = []
    vcount, fcount, ecount = [int(x) for x in file.readline().split()]
    verts = []
    facets = []
    edges = []
    i=0;
    while i<vcount:
        line = file.readline()
        if line.isspace():
            continue    # skip empty lines
        try:
             bits = [float(x) for x in line.split()]
             px = bits[0]
             py = bits[1]
             pz = bits[2]
             if use_colors:
                 colors.append([float(bits[3]) / 255, float(bits[4]) / 255, float(bits[5]) / 255])

        except ValueError:
            i=i+1
            continue
        verts.append((px, py, pz))
        i=i+1

    i=0;
    while i<fcount:
        line = file.readline()
        if line.isspace():
            continue    # skip empty lines
        try:
            splitted  = line.split()
            ids   = list(map(int, splitted))
            if len(ids) > 3:
                facets.append(tuple(ids[1:]))
            elif len(ids) == 3:
                edges.append(tuple(ids[1:]))
        except ValueError:
            i=i+1
            continue
        i=i+1

    # Assemble mesh
    off_name = bpy.path.display_name_from_filepath(filepath)
    mesh = bpy.data.meshes.new(name=off_name)
    mesh.from_pydata(verts,edges,facets)
    # mesh.vertices.add(len(verts))
    # mesh.vertices.foreach_set("co", unpack_list(verts))

    # mesh.faces.add(len(facets))
    # mesh.faces.foreach_set("vertices", unpack_face_list(facets))

    mesh.validate()
    mesh.update()

    if use_colors:
        color_data = mesh.vertex_colors.new()
        for i, facet in enumerate(mesh.polygons):
            for j, vidx in enumerate(facet.vertices):
                color_data.data[3*i + j].color = colors[vidx]

    return mesh

def save(operator, context, filepath,
    global_matrix = None,
    use_colors = False):
    # Export the selected mesh
    APPLY_MODIFIERS = True # TODO: Make this configurable
    if global_matrix is None:
        global_matrix = mathutils.Matrix()
    scene = context.scene
    obj = scene.objects.active
    mesh = obj.to_mesh(scene, APPLY_MODIFIERS, 'PREVIEW')

    # Apply the inverse transformation
    obj_mat = obj.matrix_world
    mesh.transform(global_matrix * obj_mat)

    verts = mesh.vertices[:]
    facets = [ f for f in mesh.tessfaces ]
    # Collect colors by vertex id
    colors = False
    vertex_colors = None
    if use_colors:
        colors = mesh.tessface_vertex_colors.active
    if colors:
        colors = colors.data
        vertex_colors = {}
        for i, facet in enumerate(mesh.tessfaces):
            color = colors[i]
            color = color.color1[:], color.color2[:], color.color3[:], color.color4[:]
            for j, vidx in enumerate(facet.vertices):
                if vidx not in vertex_colors:
                    vertex_colors[vidx] = (int(color[j][0] * 255.0),
                                            int(color[j][1] * 255.0),
                                            int(color[j][2] * 255.0))
    else:
        use_colors = False

    # Write geometry to file
    filepath = os.fsencode(filepath)
    fp = open(filepath, 'w')

    if use_colors:
        fp.write('COFF\n')
    else:
        fp.write('OFF\n')

    fp.write('%d %d 0\n' % (len(verts), len(facets)))

    for i, vert in enumerate(mesh.vertices):
        fp.write('%.16f %.16f %.16f' % vert.co[:])
        if use_colors:
            fp.write(' %d %d %d 255' % vertex_colors[i])
        fp.write('\n')

    #for facet in facets:
    for i, facet in enumerate(mesh.tessfaces):
        fp.write('%d' % len(facet.vertices))
        for vid in facet.vertices:
            fp.write(' %d' % vid)
        fp.write('\n')

    fp.close()

    return {'FINISHED'}

def initialize(width=512, height=448):
    """
    Setup scene, camer and lighting.

    :param width: width of rendered image
    :param height: height of rendered image
    :return: camera target
    """

    # First, the base meshes (sphere and cube) are set,
    # these are later used to display point clouds or occupancy grids.
    bpy.ops.mesh.primitive_ico_sphere_add()
    global sphere_base_mesh
    sphere_base_mesh = bpy.context.scene.objects.active.data.copy()
    for face in sphere_base_mesh.polygons:
        face.use_smooth = True

    bpy.ops.mesh.primitive_cube_add()
    global cube_base_mesh
    cube_base_mesh = bpy.context.scene.objects.active.data.copy()

    # Delete current scene, except for the camera and the lamp
    for obj in bpy.data.objects:
        if str(obj.name) in ['Camera']:
            continue
        obj.select = True
        bpy.ops.object.delete()

    scene = bpy.context.scene

    # Setup the camera, location can also be influenced later,
    # these are only defaults.
    cam = scene.objects['Camera']
    cam.location = (0, 3.0, 1.0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    def parent_obj_to_camera(b_camera):
        """
        Utility function defining the target of the camera as the origin.

        :param b_camera: camera object
        :return: origin object
        """

        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new('Empty', None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.objects.link(b_empty)
        scn.objects.active = b_empty
        return b_empty

    # Sets up the camera and defines its target.
    camera_target = parent_obj_to_camera(cam)
    cam_constraint.target = camera_target

    # For nicer visualization, several light locations are defined.
    # See the documentation for details, these should be edited based
    # on preferences.
    locations = [
        (-0.98382, 0.445997, 0.526505),
        (-0.421806, -0.870784, 0.524944),
        (0.075576, -0.960128, 0.816464),
        (0.493553, -0.57716, 0.928208),
        (0.787275, -0.256822, 0.635172),
        (1.01032, 0.148764, 0.335078)
    ]

    for i in range(len(locations)):
        # We only use point spot lamps centered at the given locations
        # and without any specific rotation (see euler angles below).
        lamp_data = bpy.data.lamps.new(name='Point Lamp ' + str(i), type='POINT')
        lamp_data.shadow_method = 'RAY_SHADOW'
        lamp_data.shadow_ray_sample_method = 'CONSTANT_QMC'
        lamp_data.use_shadow = True
        lamp_data.shadow_soft_size = 1e6
        lamp_data.distance = 2
        lamp_data.energy = 0.1
        lamp_data.use_diffuse = True
        lamp_data.use_specular = True
        lamp_data.falloff_type = 'CONSTANT'

        lamp_object = bpy.data.objects.new(name='Spot Lamp ' + str(i), object_data=lamp_data)
        scene.objects.link(lamp_object)
        lamp_object.location[0] = locations[i][0]
        lamp_object.location[1] = locations[i][1]
        lamp_object.location[2] = locations[i][2]
        lamp_object.rotation_euler[0] = 0
        lamp_object.rotation_euler[1] = 0
        lamp_object.rotation_euler[2] = 0

        lamp_object.parent = camera_target

    # This tries to use CUDA rendering if possible.
    try:
        if (2, 78, 0) <= bpy.app.version:
            # https://blender.stackexchange.com/questions/5281/blender-sets-compute-device-cuda-but-doesnt-use-it-for-actual-render-on-ec2
            bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        else:
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    except TypeError:
        pass

    scene.render.use_file_extension = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_antialiasing = True
    scene.render.use_shadows = True
    world = bpy.context.scene.world
    world.zenith_color = [1.0, 1.0, 1.0]
    world.horizon_color = [1.0, 1.0, 1.0]
    scene.render.alpha_mode = 'SKY'
    world.light_settings.use_environment_light = True
    world.light_settings.environment_color = 'PLAIN'
    world.light_settings.environment_energy = 0.5

    return camera_target


def make_material(name, diffuse, alpha, shadow=False):
    """
    Creates a material with the given diffuse and alpha. If shadow is true the
    object casts and receives shadows.

    :param name: name of material
    :param diffuse: diffuse color (in rgb)
    :param alpha: alpha (float in [0,1])
    :param shadow: whether to cast/receive shadows
    :return: material
    """

    material = bpy.data.materials.new(name)
    material.diffuse_color = diffuse
    material.diffuse_shader = 'LAMBERT'
    material.diffuse_intensity = 1
    material.specular_color = (1, 1, 1)
    material.specular_shader = 'COOKTORR'
    material.specular_intensity = 2
    material.alpha = alpha
    material.use_transparency = True
    material.ambient = 1.0

    material.use_cast_shadows = shadow
    material.use_shadows = shadow

    return material


def load_off(off_file, material, offset=(0, 0, 0), scale=1, axes='xyz'):
    """
    Loads a triangular mesh from an OFF file. For pre-processing, mesh.py can be used;
    the function still allows to define an offset (to translate the mesh) and a scale.

    The axes parameter defines the order of the axes. Using xzy, for example, assumes
    that the first coordinate is x, the second is z and the third is y.

    **Note that first, the axes are swapper, then the OFF is scaled and finally translated.**

    :param off_file: path to OFF file
    :param material: previously defined material
    :param offset: offset after scaling
    :param scale: scaling
    :param axes: axes definition
    """

    # This used import_off.py, see README for license.
    bpy.ops.import_mesh.off(filepath=off_file)

    assert len(offset) == 3
    assert scale > 0
    assert len(axes) == 3

    x_index = axes.find('x')
    y_index = axes.find('y')
    z_index = axes.find('z')

    assert x_index >= 0 and x_index < 3
    assert y_index >= 0 and y_index < 3
    assert z_index >= 0 and z_index < 3
    assert x_index != y_index and x_index != z_index and y_index != z_index

    for obj in bpy.context.scene.objects:

        # obj.name contains the group name of a group of faces, see http://paulbourke.net/dataformats/obj/
        # every mesh is of type 'MESH', this works not only for ShapeNet but also for 'simple'
        # obj files
        if obj.type == 'MESH' and not 'BRC' in obj.name:

            # change color
            # this is based on https://stackoverflow.com/questions/4644650/blender-how-do-i-add-a-color-to-an-object
            # but needed changing a lot of attributes according to documentation
            obj.data.materials.append(material)

            for vertex in obj.data.vertices:
                # make a copy, otherwise axes switching does not work
                vertex_copy = (vertex.co[0], vertex.co[1], vertex.co[2])

                # First, swap the axes, then scale and offset.
                vertex.co[0] = vertex_copy[x_index]
                vertex.co[1] = vertex_copy[y_index]
                vertex.co[2] = vertex_copy[z_index]

                vertex.co[0] = vertex.co[0] * scale + offset[0]
                vertex.co[1] = vertex.co[1] * scale + offset[1]
                vertex.co[2] = vertex.co[2] * scale + offset[2]

            obj.name = 'BRC_' + obj.name


def load_txt(txt_file, radius, material, offset=(0, 0, 0), scale=1, axes='xyz'):
    """
    Load a point cloud from txt file, see the documentation for the format.
    Additionally, the radius of the points, an offset and a scale can be defined, for details
    on the parameters also see load_off.

    :param txt_file: path to TXT file
    :param radius: radius of rendered points/spheres
    :param material: previously defined material
    :param offset: offset
    :param scale: scale
    :param axes: axes definition
    :return:
    """

    global sphere_base_mesh

    assert len(offset) == 3
    assert scale > 0
    assert len(axes) == 3

    x_index = axes.find('x')
    y_index = axes.find('y')
    z_index = axes.find('z')

    assert x_index >= 0 and x_index < 3
    assert y_index >= 0 and y_index < 3
    assert z_index >= 0 and z_index < 3
    assert x_index != y_index and x_index != z_index and y_index != z_index

    voxel_file = open(txt_file, 'r')
    voxel_lines = voxel_file.readlines()
    voxel_file.close()

    mesh = bmesh.new()
    for line in voxel_lines:
        vals = line.split(' ')
        if not line.startswith('#') and line.strip() != '' and len(vals) >= 3:
            location = (
                float(vals[x_index]) * scale + offset[0],
                float(vals[y_index]) * scale + offset[1],
                float(vals[z_index]) * scale + offset[2]
            )

            m = sphere_base_mesh.copy()
            for vertex in m.vertices:
                vertex.co[0] = vertex.co[0] * radius + location[0]
                vertex.co[1] = vertex.co[1] * radius + location[1]
                vertex.co[2] = vertex.co[2] * radius + location[2]

            mesh.from_mesh(m)

    mesh2 = bpy.data.meshes.new('Mesh')
    mesh.to_mesh(mesh2)

    obj = bpy.data.objects.new('BRC_Point_Cloud', mesh2)
    obj.data.materials.append(material)
    obj.active_material_index = 0
    obj.active_material = material

    bpy.context.scene.objects.link(obj)


def load_binvox(binvox_file, radius, material, offset, scale, axes):
    """
    Load a binvox file, see binvox_rw.py for format. Again, radius of the cubes, material, offset and scale
    can be defined as in load_off.

    :param binvox_file: path to binvox file
    :param radius: radius, i.e. side length, of cubes
    :param material: previously defined material
    :param offset: offset
    :param scale: scale
    :param axes: axes definition
    :return:
    """

    global cube_base_mesh

    assert len(offset) == 3
    assert len(scale) == 3
    assert len(axes) == 3

    x_index = axes.find("x")
    y_index = axes.find("y")
    z_index = axes.find("z")

    assert x_index >= 0 and x_index < 3
    assert y_index >= 0 and y_index < 3
    assert z_index >= 0 and z_index < 3
    assert x_index != y_index and x_index != z_index and y_index != z_index

    with open(binvox_file, 'rb') as f:
        model = read_as_3d_array(f)

    points = np.where(model.data)
    locations = np.zeros((points[0].shape[0], 3), dtype=float)
    locations[:, 0] = (points[x_index][:] + 0.5) / model.data.shape[x_index]
    locations[:, 1] = (points[y_index][:] + 0.5) / model.data.shape[y_index]
    locations[:, 2] = (points[z_index][:] + 0.5) / model.data.shape[z_index]
    locations[:, 0] -= 0.5
    locations[:, 1] -= 0.5
    locations[:, 2] -= 0.5

    locations[:, 0] = locations[:, 0] * scale[0] + offset[0]
    locations[:, 1] = locations[:, 1] * scale[1] + offset[1]
    locations[:, 2] = locations[:, 2] * scale[2] + offset[2]

    mesh = bmesh.new()
    for i in range(locations.shape[0]):
            m = cube_base_mesh.copy()
            for vertex in m.vertices:
                vertex.co[0] = vertex.co[0] * radius + locations[i, 0]
                vertex.co[1] = vertex.co[1] * radius + locations[i, 1]
                vertex.co[2] = vertex.co[2] * radius + locations[i, 2]

            mesh.from_mesh(m)

    mesh2 = bpy.data.meshes.new('Mesh')
    mesh.to_mesh(mesh2)

    obj = bpy.data.objects.new('BRC_Occupancy', mesh2)
    obj.data.materials.append(material)
    obj.active_material_index = 0
    obj.active_material = material

    bpy.context.scene.objects.link(obj)


def render(camera_target, output_file, rotation, distance):
    """
    Render all loaded objects into the given object files. Additionally, the
    rotation of the camera around the origin and the distance can be defined.

    The first argument is the camera_target returned from initialize().

    :param camera_target: returned by initialize()
    :param output_file: path to output file
    :param rotation: rotation of camera
    :param distance: distance to target
    """

    bpy.context.scene.render.filepath = output_file

    camera_target.rotation_euler[0] = math.radians(rotation[0])
    camera_target.rotation_euler[1] = math.radians(rotation[1])
    camera_target.rotation_euler[2] = math.radians(rotation[2])

    cam = bpy.context.scene.objects['Camera']
    cam.location = (0, 3.0 * distance, 1.0 * distance)

    bpy.ops.render.render(animation=False, write_still=True)

def main():
    """
    Main function for rendering a specific binvox file.
    """

    parser = argparse.ArgumentParser(description='Renders an occupancy grid (BINVOX file).')
    parser.add_argument('--binvox', type=str, help='Path to OFF file.')
    parser.add_argument('--output', type=str, default='output.png', help='Path to output PNG image.')

    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        argv = ""
    args = parser.parse_args(argv)

    if not os.path.exists(args.binvox):
        log('BINVOX file not found.', LogLevel.ERROR)
        exit()

    camera_target = initialize()
    binvox_material = make_material('BRC_Material_Occupancy', (1.0, 1.0, 1.0), 0.8, True)

    load_binvox(args.binvox, 0.0125, binvox_material, (0, 0, 0), (1, 1, 1), 'xzy')#umiformer lrgt


    rotation = (0, 0, -135)
    distance = 0.6
    #3dswin,3dretr chair
    #rotation = (20
    #            ,0, 90)
    #3dswin_real chair
    #rotation = (0
    #            ,0, 0)
    #3dsiwn_longchair
    #rotation = (0
    #            ,0,-135)
    #rotation = (0
     #           ,180,0)
    #distance = 0.6
    render(camera_target, args.output, rotation, distance)


if __name__ == '__main__':
    main()
