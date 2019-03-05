# TODO write function that takes verts and camera params as input and goes to bodypart seg
# can then implement this as Keras lambda layer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
from keras import backend as K


def projects_to_seg(input, img_wh):
    """

    :param projects:
    :return:
    """
    projects_with_depth, mask_vals = input
    projects = projects_with_depth[:, :, :2]

    part_indices_path = "./keras_smpl/part_vertices.pkl"

    with open(part_indices_path, 'rb') as f:
        part_indices = pickle.load(f)

    i = tf.range(0, img_wh)
    j = tf.range(0, img_wh)

    t1, t2 = tf.meshgrid(i, j)
    grid = tf.cast(tf.stack([t1, t2], axis=2), dtype='float32')  # img_wh x img_wh x 2
    reshaped_grid = tf.reshape(grid, [-1, 2])  # img_wh^2 x 2

    segs = []
    for part in range(len(part_indices)):
        indices = part_indices[part]
        num_indices = len(indices)
        indices = tf.constant(indices, dtype='int32')

        part_projects = tf.gather(projects, indices, axis=1)  # N x num_indices x 2
        part_projects = tf.tile(tf.expand_dims(part_projects, axis=1),
                                [1, img_wh*img_wh, 1, 1])  # N x img_wh^2 x num_indices x 2

        part_mask_vals = tf.gather(mask_vals, indices, axis=1) # N x num_indices
        part_mask_vals = tf.tile(tf.expand_dims(part_mask_vals, axis=1),
                                 [1, img_wh*img_wh, 1])  # N x img_wh^2 x num_indices

        expanded_grid = tf.tile(tf.expand_dims(reshaped_grid, axis=1),
                                [1, num_indices, 1])  # img_wh^2 x num_indices x 2

        diff = tf.subtract(part_projects, expanded_grid)  # N x img_wh^2 x num_indices x 2
        norm = tf.norm(diff, axis=3)  # N x img_wh^2 x num_indices
        norm = tf.multiply(norm, part_mask_vals)
        exp = tf.exp(tf.negative(norm))  # N x img_wh^2 x num_indices
        scores = tf.reduce_max(exp, axis=2)  # N x img_wh^2
        seg = tf.reshape(scores, [-1, img_wh, img_wh])  # N x img_wh x img_wh
        segs.append(seg)

    stacked_segs = tf.stack(segs, axis=3)  # N x img_wh x img_wh x 31
    silhouettes = tf.subtract(1.0,
                              tf.clip_by_value(tf.reduce_sum(stacked_segs, axis=3),
                                               clip_value_min=0,
                                               clip_value_max=1))  # N x img_wh x img_wh
    # segs.insert(0, silhouettes)
    output_segs = tf.concat([tf.expand_dims(silhouettes, axis=3), stacked_segs],
                            axis=3)  # N x img_wh x img_wh x 32
    output_segs = tf.reverse(output_segs, axis=[1])  # Flip image vertically
    return output_segs



