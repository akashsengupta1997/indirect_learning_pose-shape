
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import backend as K


def persepective_project(verts):
    """
    INCOMPLETE - I don't use this anywhere.
    Perspective projection of vertices onto image plane.
    For now, the camera parameters (T and intrinsics) are hard-coded here - could try and learn
    them in the future?
    :param verts: Batch of vertices with shape N x 6890 x 3
    :return:
    """
    img_wh = 64

    # k_u = smpl[:, 0]
    # k_v = smpl[:, 1]
    # T1 = smpl[:, 2]
    # T2 = smpl[:, 3]
    # T3 = smpl[:, 4]

    k_u = 200.0
    k_v = 200.0
    T1 = tf.expand_dims(tf.constant(0.0), axis=0)
    T2 = tf.expand_dims(tf.constant(0.0), axis=0)
    T3 = tf.expand_dims(tf.constant(10.0), axis=0)

    # Rigid body transformation
    T = tf.stack([T1, T2, T3], axis=1)
    T = tf.expand_dims(T, axis=1)
    verts = tf.add(T, verts)

    # Perspective Projection
    x = tf.div(verts[:, :, 0], verts[:, :, 2])
    y = tf.div(verts[:, :, 1], verts[:, :, 2])
    u0 = img_wh / 2.0
    v0 = img_wh / 2.0
    # k_u = tf.tile(tf.expand_dims(k_u, axis=1), [1, 6890])
    # k_v = tf.tile(tf.expand_dims(k_v, axis=1), [1, 6890])
    # u = tf.add(u0, tf.multiply(x, k_u))
    # v = tf.add(v0, tf.multiply(y, k_v))
    u = tf.add(u0, tf.scalar_mul(k_u, x))
    v = tf.add(v0, tf.scalar_mul(k_v, y))
    project_coords = tf.stack([u, v], axis=2)

    return project_coords


def orthographic_project(inputs, vertex_sampling):
    """
    Orthographically project, scale and translate 3D vertices on 2D image plane.
    :param inputs: 3D vertex mesh (6890 3D vertices) and camera parameters.
    :return: scaled and translated 2D vertices along with their original depth values (needed
    for visibility mask computation).
    """
    verts, smpl = inputs
    k_u = smpl[:, 0]
    k_v = smpl[:, 1]
    u0 = smpl[:, 2]
    v0 = smpl[:, 3]

    if vertex_sampling is not None:
        verts = verts[:, ::vertex_sampling, :]

    x_proj = verts[:, :, 0]
    y_proj = verts[:, :, 1]
    z = verts[:, :, 2]
    k_u = tf.tile(tf.expand_dims(k_u, axis=1), [1, K.shape(verts)[1]])
    k_v = tf.tile(tf.expand_dims(k_v, axis=1), [1, K.shape(verts)[1]])
    u0 = tf.tile(tf.expand_dims(u0, axis=1), [1, K.shape(verts)[1]])
    v0 = tf.tile(tf.expand_dims(v0, axis=1), [1, K.shape(verts)[1]])
    u = tf.add(u0, tf.multiply(x_proj, k_u))
    v = tf.add(v0, tf.multiply(y_proj, k_v))
    project_coords_with_depth = tf.stack([u, v, z], axis=2)

    return project_coords_with_depth
