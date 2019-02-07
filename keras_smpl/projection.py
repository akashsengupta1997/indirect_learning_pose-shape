
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def persepective_project(inputs):
    """
    :param X: N x num_vertices x 3 tensor
    :param camera: N x num_camera_params
    :return:
    """
    verts = inputs[0]
    smpl = inputs[1]
    img_wh = 64

    k_u = smpl[:, 0]
    k_v = smpl[:, 1]
    T1 = smpl[:, 2]
    T2 = smpl[:, 3]
    T3 = smpl[:, 4]
    print('Cam params: ku, kv', k_u.shape, k_v.shape, T1.shape, T2.shape, T3.shape)

    # Rigid body transformation
    T = tf.stack([T1, T2, T3], axis=1)
    T = tf.expand_dims(T, axis=1)
    verts = tf.add(T, verts)

    # Perspective Projection
    x = tf.div(verts[:, :, 0], verts[:, :, 2])
    y = tf.div(verts[:, :, 1], verts[:, :, 2])
    u0 = img_wh / 2.0
    v0 = img_wh / 2.0
    k_u = tf.tile(tf.expand_dims(k_u, axis=1), [1, 6890])
    k_v = tf.tile(tf.expand_dims(k_v, axis=1), [1, 6890])
    u = tf.add(u0, tf.multiply(x, k_u))
    v = tf.add(v0, tf.multiply(y, k_v))
    pixel_coords = tf.stack([u, v], axis=2)

    return pixel_coords
