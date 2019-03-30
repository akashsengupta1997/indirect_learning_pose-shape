
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from keras import backend as K


def compute_mask_batch_map_only(batch_projects_with_depth):
    """
    Outputs N x num_vertices tensor of mask_vals: mask_val is either 500 (if vertex is invisible) or
    1 if visible.
    During projections to segmentation stage layer (projects_to_seg.py), this mask will be used
    to ensure that invisible vertices do not affect the segmentation output.
    :param batch_projects_with_depth: N x num_vertices x 3 tensor of projected vertices and their
    original depth values i.e. (u, v, z)
    :return:
    """
    batch_pixels = tf.round(batch_projects_with_depth[:, :, :2])  # N x num_vertices x 2
    batch_pixels_with_depth = tf.concat([batch_pixels,
                                         tf.expand_dims(batch_projects_with_depth[:, :, 2], axis=2)],
                                        axis=2)  # N x num_vertices x 3

    masks = tf.map_fn(compute_mask_map_over_batch,
                      batch_pixels_with_depth,
                      dtype='float32',
                      back_prop=False)  # N x num_vertices

    return masks


def compute_mask_map_over_batch(pixels_with_depth):
    """
    Applied to each member of the batch in batch_pixels_with_depth.
    It outputs a mask tensor with shape (num_vertices,).
    :param pixels_with_depth: num_vertices x 3 tensor of rounded + projected vertices and their
    original depth values.
    :return: mask
    """
    img_wh = 48
    num_pixels = pixels_with_depth.get_shape().as_list()[0]  # num_vertices
    indices = tf.expand_dims(tf.range(num_pixels, dtype='float32'), axis=1)  # num_vertices x 1
    pixels_with_depth_and_index = tf.concat([pixels_with_depth, indices], axis=1)  # num_vertices x 4

    # ---TESTING for loop---
    min_indices = []
    for i in range(img_wh):
        print('Row', i)
        for j in range(img_wh):
            pixel_coord = tf.constant([i, j], dtype='float32')
            num_pixels = pixels_with_depth_and_index.get_shape().as_list()[0]
            pixel_coord = tf.tile(tf.expand_dims(pixel_coord, axis=0), [num_pixels, 1],
                                  name='big_tile2')  # num_vertices x 2

            vert_indices_at_pixel = tf.where(tf.reduce_all(tf.equal(pixel_coord,
                                                                    pixels_with_depth_and_index[
                                                                    :, :2]),
                                                           axis=1))  # ? x 1

            verts_at_pixel = tf.gather(pixels_with_depth_and_index,
                                       vert_indices_at_pixel)  # ? x 1 x 4
            vert_depths_at_pixel = verts_at_pixel[:, :, 2]  # ? x 1
            is_empty = tf.equal(tf.size(vert_depths_at_pixel), tf.constant(0))

            min_depth_vert_at_pixel = tf.cond(is_empty,
                                              lambda: tf.ones([1, 1, 4]),
                                              lambda: tf.gather(verts_at_pixel,
                                                                tf.argmax(vert_depths_at_pixel,
                                                                          axis=0)),
                                              )  # 1 x 1 x 4

            min_depth_vert_index_at_pixel = tf.squeeze(tf.cast(min_depth_vert_at_pixel[:, :, 3],
                                                               dtype='int32'),
                                                       axis=1)  # (1,)
            min_indices.append(min_depth_vert_index_at_pixel)

    min_indices = tf.concat(min_indices, axis=0)
    min_indices, _ = tf.unique(min_indices)

    mask = K.variable(np.ones(num_pixels) * 500)
    ones = tf.ones_like(min_indices, dtype='float32')
    mask = tf.scatter_update(mask, min_indices, ones)  # (num_vertices,)

    return mask