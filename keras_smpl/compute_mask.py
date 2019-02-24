
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from keras import backend as K


def compute_mask(batch_projects_with_depth):
    """
    Outputs N x 6890 tensor of mask_vals: mask_val is either 100 (if vertex is invisible) or
    1 if visible.
    :param batch_projects_with_depth: N x 6890 x 3 tensor of projected vertices and their
    original depth values i.e. (u, v, z)
    :return:
    """
    batch_pixels = tf.round(batch_projects_with_depth[:, :, :2])  # N x 6890 x 2
    batch_pixels_with_depth = tf.concat([batch_pixels,
                                         tf.expand_dims(batch_projects_with_depth[:, :, 2], axis=2)],
                                        axis=2)  # N x 6890 x 3

    masks = tf.map_fn(compute_mask_map_over_batch,
                      batch_pixels_with_depth,
                      dtype='float32',
                      back_prop=False)

    return masks


def compute_mask_map_over_batch(pixels_with_depth):
    """

    :param pixels_with_depth: 6890 x 3
    :return:
    """
    img_wh = 64
    indices = tf.expand_dims(tf.range(6890, dtype='float32'), axis=1)
    pixels_with_depth_and_index = tf.concat([pixels_with_depth, indices], axis=1)  # 6890 x 4

    i = tf.range(0, img_wh)
    j = tf.range(0, img_wh)

    t1, t2 = tf.meshgrid(i, j)
    grid = tf.cast(tf.stack([t1, t2], axis=2), dtype='float32')  # img_wh x img_wh x 2
    pixel_coords = tf.reshape(grid, [-1, 2])  # img_wh^2 x 2
    expanded_pixels_with_depth_and_index = tf.tile(tf.expand_dims(pixels_with_depth_and_index,
                                                                  axis=0),
                                                   [img_wh*img_wh, 1, 1]) # img_wh^2 x 6890 x 4
    min_indices = tf.map_fn(get_min_depth_vert_index_at_pixel,
                            [pixel_coords, expanded_pixels_with_depth_and_index],
                            back_prop=False,
                            dtype='int32')
    min_indices, _ = tf.unique(tf.squeeze(min_indices))

    mask = K.variable(np.ones((6890))*50)
    ones = tf.ones_like(min_indices, dtype='float32')
    mask = tf.scatter_update(mask, min_indices, ones)

    return mask


def get_min_depth_vert_index_at_pixel(input):
    pixel_coord, pixels_with_depth_and_index = input
    pixel_coord = tf.tile(tf.expand_dims(pixel_coord, axis=0), [6890, 1])  # 6890 x 2

    vert_indices_at_pixel = tf.where(tf.reduce_all(tf.equal(pixel_coord,
                                                            pixels_with_depth_and_index[:, :2]),
                                                   axis=1))

    verts_at_pixel = tf.gather(pixels_with_depth_and_index, vert_indices_at_pixel)  # n x 1 x 4
    vert_depths_at_pixel = verts_at_pixel[:, :, 2]  # n x 1
    is_empty = tf.equal(tf.size(vert_depths_at_pixel), tf.constant(0))

    min_depth_vert_at_pixel = tf.cond(is_empty,
                                      lambda: tf.ones([1, 1, 4]),
                                      lambda: tf.gather(verts_at_pixel,
                                                        tf.argmax(vert_depths_at_pixel,
                                                                  axis=0)),
                                      )  # 1 x 1 x 4
    # Just returning ones if verts_at_pixel is empty is a pretty hacky way of getting
    # around argmin-ing a possibly empty tensor

    min_depth_vert_index_at_pixel = tf.squeeze(tf.cast(min_depth_vert_at_pixel[:, :, 3],
                                                        dtype='int32'),
                                               axis=1)  # (1,)
    return min_depth_vert_index_at_pixel
