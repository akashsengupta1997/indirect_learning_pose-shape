
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from keras import backend as K


def compute_mask(batch_projects_with_depth):
    """
    Outputs N x num_vertices tensor of mask_vals: mask_val is either 500 (if vertex is
    invisible) or 1 if visible.
    During projections to segmentation stage layer (projects_to_seg.py), this mask will be used
    to ensure that invisible vertices do not affect the segmentation output.
    :param batch_projects_with_depth: N x num_vertices x 3 tensor of projected vertices and
    their original depth values i.e. (u, v, z)
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
    Used in the map_fn call in compute_mask.
    Applied to each member of the batch in batch_pixels_with_depth.
    It outputs a mask tensor with shape (num_vertices,).
    :param pixels_with_depth: num_vertices x 3 tensor of rounded + projected vertices and their
    original depth values.
    :return: mask
    """
    img_wh = 64
    num_vertices = pixels_with_depth.get_shape().as_list()[0]  # num_vertices
    indices = tf.expand_dims(tf.range(num_vertices, dtype='float32'), axis=1)  # num_vertices x 1
    pixels_with_depth_and_index = tf.concat([pixels_with_depth, indices], axis=1)  # num_vertices x 4

    i = tf.range(0, img_wh)
    j = tf.range(0, img_wh)

    t1, t2 = tf.meshgrid(i, j)
    grid = tf.cast(tf.stack([t1, t2], axis=2), dtype='float32')  # img_wh x img_wh x 2
    pixel_coords = tf.reshape(grid, [-1, 2])  # img_wh^2 x 2
    expanded_pixels_with_depth_and_index = tf.tile(tf.expand_dims(pixels_with_depth_and_index,
                                                                  axis=0, name='big_tile1'),
                                                   [img_wh*img_wh, 1, 1])  # img_wh^2 x num_vertices x 4

    min_depth_verts = tf.map_fn(get_min_depth_vert_index_at_pixel,
                                [pixel_coords, expanded_pixels_with_depth_and_index],
                                back_prop=False,
                                dtype='float32',
                                parallel_iterations=128)  # img_wh^2 x 1 x 1 x 4

    min_depth_verts = tf.squeeze(tf.cast(min_depth_verts, dtype='int32'))  # img_wh^2 x 4
    min_indices, _ = tf.unique(min_depth_verts[:, 3])  # (?,), ? is number of visible vertices

    mask = K.variable(np.ones(num_vertices) * 500)
    ones = tf.ones_like(min_indices, dtype='float32')
    mask = tf.scatter_update(mask, min_indices, ones)  # (num_vertices,)

    return mask


def get_min_depth_vert_index_at_pixel(input):
    """
    Used in the map_fn call in  compute_mask_map_over_batch.
    Outputs the index of the minimum depth (foreground) pixel at each location specified in
    pixel_coords.
    :param input: [pixel_coords,  expanded_pixels_with_depth_and_index]
    pixel_coords is tensor of locations of all image pixels (i.e. (0,0) to (img_wh-1, img_wh-1)
    pixels_with_depth_and_index is tensor of all rounded + projected vertices, concatenated
    with their original depth and index number.
    :return: min_depth_vert_index_at_pixel
    """
    pixel_coord, pixels_with_depth_and_index = input
    num_vertices = pixels_with_depth_and_index.get_shape().as_list()[0]
    pixel_coord = tf.tile(tf.expand_dims(pixel_coord, axis=0), [num_vertices, 1], name='big_tile2')  # num_vertices x 2

    vert_indices_at_pixel = tf.where(tf.reduce_all(tf.equal(pixel_coord,
                                                            pixels_with_depth_and_index[:, :2]),
                                                   axis=1))  # ? x 1

    verts_at_pixel = tf.gather(pixels_with_depth_and_index, vert_indices_at_pixel)  # ? x 1 x 4
    vert_depths_at_pixel = verts_at_pixel[:, :, 2]  # ? x 1
    is_empty = tf.equal(tf.size(vert_depths_at_pixel), tf.constant(0))

    min_depth_vert_at_pixel = tf.cond(is_empty,
                                      lambda: tf.ones([1, 1, 4]),
                                      lambda: tf.gather(verts_at_pixel,
                                                        tf.argmax(vert_depths_at_pixel,
                                                                  axis=0)),
                                      )  # 1 x 1 x 4
    # Just returning ones if verts_at_pixel is empty is a pretty hacky way of getting
    # around argmin-ing a possibly empty tensor - but I need to do it  because tf.cond must
    # output something for if and else conditions.

    return min_depth_vert_at_pixel
