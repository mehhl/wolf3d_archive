""" This is a code to re-skin the original dm_control's Dog model using a custom mesh.
    The code tries to adjust for some of the transformation Blender does to stl's it reads
    so that it is possible to build a new mesh based on the original dog .stl file. """

import numpy as np
import pywavefront
from stl import mesh
from scipy.spatial import cKDTree

from dm_control.utils import io as resources
from dm_control.mjcf import skin

def read_skin(fp):
    """Reads MuJoCo's .skn file."""
    class FakeBody:
        def __init__(self, full_identifier):
            self.full_identifier = full_identifier
    raw_skn = resources.GetResource(src_fp, mode='r+b')
    src_skn = skin.parse(raw_skn, body_getter=FakeBody)
    return src_skn
 

def read_blender_obj(fp):
    """Reads .obj file from Blender and reverses Blender's coordinate transform.
    Returns: read obj and transformed vertices/faces as a numpy array."""
    obj = pywavefront.Wavefront(fp, collect_faces=True)
    transform = np.array(obj.vertices, dtype=np.float32)
    transform[:, [1, 2]] = transform[:, [2, 1]]
    transform[:, 1] *= -1
    faces = np.array(obj.meshes['SKINbody'].faces, dtype=np.int32)
    return transform, faces
 

def list_texcoords_and_faces(fp):
    """List texcoords and texturized faces from an .obj stored in path fp."""
    texcoords = []
    texfaces = []
    with open(fp, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'vt':
                texcoords.append(list(map(float, parts[1:])))
            if parts[0] == 'f':
                texfaces.append(list(map(str, parts[1:])))
    return texcoords, texfaces
 

def calculate_target_weights(src_skn, transform_vertices, src_vertices, k):
    """Calculate target weights for the vertices using kNN interpolation."""
    src_weight_matrix = np.zeros((len(src_skn.vertices), len(src_skn.bones)), 
                                 dtype=np.float32)
    for j, bone in enumerate(src_skn.bones):
        for vert_idx, weight in zip(bone.vertex_ids, bone.vertex_weights):
            src_weight_matrix[vert_idx, j] = weight
 
    tree = cKDTree(src_skn.vertices)
    distances, nearest_idxs = tree.query(src_vertices, k=1)
    if (k > 1):
        weights = 1.0 / distances
        weights /= np.sum(weights, axis=1, keepdims=True)

    target_weight_matrix = np.zeros((len(src_vertices), len(src_skn.bones)), 
                                    dtype=np.float32)
    for i in range(len(src_vertices)):
        if (k > 1):
            for j in range(k):
                target_weight_matrix[i, :] += weights[i, j] * \
                        src_weight_matrix[nearest_idxs[i, j], :]
        else:
            target_weight_matrix[i, :] = src_weight_matrix[nearest_idxs[i], :]
    return target_weight_matrix
 

def calculate_texcoords(orig_texcoords, orig_vertices, src_vertices):
    """Performs kNN interpolation to calculate texcoordinates for src_vertices
    from orig_texcoords via orig_vertices."""
    tree = cKDTree(orig_vertices)
    _, nearest_idxs = tree.query(src_vertices, k=1)
    src_texcoords = []
    for i, vertex in enumerate(src_vertices):
        src_texcoords.append(orig_texcoords[nearest_idxs[i]])
    return np.array(src_texcoords)
  

# --------------------------------------

src_fp = '../dog_skin.skn'
transform_fp = '../dog_original.obj'
obj_fp = '../dog_sculptI.obj'
target_fp = '../dog_sculptI.skn'

src_skn = read_skin(src_fp)
transform_vertices, transform_faces = read_blender_obj(transform_fp)
src_vertices, src_faces = read_blender_obj(obj_fp)
src_faces

target_weight_matrix = calculate_target_weights(src_skn, transform_vertices, src_vertices, 1)

target_bones = []
for j, bone in enumerate(src_skn.bones):
    body = bone.body
    bindpos = bone.bindpos
    bindquat = bone.bindquat
    vertex_ids = np.arange(len(src_vertices))
    vertex_weights = target_weight_matrix[:, j]
    target_bone = skin.Bone(body, bindpos, bindquat, vertex_ids, vertex_weights)
    target_bones.append(target_bone)
bones = target_bones

orig_texcoords = src_skn.texcoords
orig_vertices = src_skn.vertices
src_texcoords = calculate_texcoords(orig_texcoords, orig_vertices, src_vertices)

target_skn = skin.Skin(src_vertices, src_texcoords, src_faces, bones)
 
# serialize the target
serialized_target_skn = skin.serialize(target_skn)
with open(target_fp, 'wb') as f:
    f.write(serialized_target_skn)
                
#### Debugging notebook

## I want to compare source skn (taken from MuJoCo) and target skn (taken from my process) to see where they differ
## the place where they differ is certain to be a reason for at least some of the errors

src_skn    # this is the source (from dm_control's github)
target_skn # this is the target (after running my process)

src_skn.faces
target_skn.faces

# element by element -- what is wrong? It turns out that there are less vertices int he target skin compared to the source skin. Why? What can be done about this?

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

src_skn.vertices[0:10]
target_skn.vertices[0:10]
src_skn.faces[0:10]
target_skn.faces[0:10]


print(target_skn.vertices)
target_skn.faces
src_skn.faces
len(target_skn.faces)
len(src_skn.faces)

dir(src_skn)
dir(src_skn.bones[0])

tree = cKDTree(target_skn.vertices)
distances, nearest_idxs = tree.query(src_skn.vertices, k=1)
nearest_idxs[0:5]
nearest_idxs[500:750]
target_skn.bones[0].vertex_ids
target_skn.bones[0].vertex_weights
src_skn.bones[0].vertex_weights
src_skn.bones[0].vertex_ids
len(src_skn.bones[0].vertex_ids)
len(target_skn.bones[0].vertex_ids)
src_skn.vertices
target_skn.vertices

len(src_skn.vertices)
len(target_skn.vertices)

sum(target_skn.bones[0].vertex_weights)
sum(src_skn.bones[0].vertex_weights)


src_skn.vertices[0:5]
target_skn.vertices[0:5]
src_skn.bones[0].vertex_ids[0:5]
src_skn.bones[0].vertex_weights[0:5]
target_skn.bones[0].vertex_ids[0:5]
target_skn.bones[0].vertex_weights[0:5]


texcoords = []
fp = obj_fp
with open(fp, 'r') as f:
    for line in f:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'vt:':
            texcoords.append(list(map(float, parts[1:])))

 


