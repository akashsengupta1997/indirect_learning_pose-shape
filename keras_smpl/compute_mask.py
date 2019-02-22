
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
    print('Masks', masks.get_shape())

    return masks


def compute_mask_map_over_batch(pixels_with_depth):
    """

    :param pixels_with_depth: 6890 x 3
    :return:
    """
    img_wh = 96
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


    # min_depth_vert_indices = []
    # for i in range(img_wh):
    #     for j in range(img_wh):
    #         pixel_coord = tf.constant([i, j], dtype='float32')
    #         pixel_coord = tf.tile(tf.expand_dims(pixel_coord, axis=0), [6890, 1])  # 6890 x 2
    #         equal = tf.equal(pixel_coord, pixels_with_depth[:, :2])  # 6890 x 2
    #         # print('equal', equal.get_shape())
    #         All = tf.reduce_all(equal, axis=1)  # 6890
    #         # print('all', All.get_shape())
    #         vert_indices_at_pixel = tf.where(All)  # n x 1, where n = num verts at pixel - can squeeze out 1 dimension
    #         # print('indices of verts at pixel', vert_indices_at_pixel.get_shape())
    #
    #         verts_at_pixel = tf.gather(pixels_with_depth_and_index,
    #                                          vert_indices_at_pixel)  # n x 1 x 4
    #         # print('coords, depth and index of verts at pixel', verts_at_pixel.get_shape())
    #         vert_depths_at_pixel = verts_at_pixel[:, :, 2]  # n x 1
    #         # print('depths of verts at pixel', vert_depths_at_pixel.get_shape())
    #
    #         is_empty = tf.equal(tf.size(vert_depths_at_pixel), tf.constant(0))
    #         # is_empty = K.print_tensor(is_empty, message='is_empty is')
    #
    #         min_depth_vert_at_pixel = tf.cond(is_empty,
    #                                           lambda: tf.ones([1, 1, 4]),
    #                                           lambda: tf.gather(verts_at_pixel,
    #                                                             tf.argmax(vert_depths_at_pixel,
    #                                                                       axis=0)),
    #                                           )  # 1 x 1 x 4
    #         # Just returning ones if verts_at_pixel is empty is a pretty hacky way of getting
    #         # around argmin-ing a possibly empty tensor
    #
    #         # print('min depth vert at pixel', min_depth_vert_at_pixel.get_shape())
    #         min_depth_vert_index_at_pixel = tf.squeeze(
    #             tf.cast(min_depth_vert_at_pixel[:, :, 3], dtype='int32'),
    #             axis=1)  # (1,)
    #         # print('min depth index at pixel', min_depth_vert_index_at_pixel.get_shape())
    #         min_depth_vert_indices.append(min_depth_vert_index_at_pixel)
    #
    # stacked_min_indices = tf.concat(min_depth_vert_indices, axis=0)
    # print('stacked indices', stacked_min_indices.get_shape())
    # stacked_min_indices, _ = tf.unique(stacked_min_indices)
    # print('stacked indices unique', stacked_min_indices.get_shape())

    mask = K.variable(np.ones((6890))*10)
    ones = tf.ones_like(min_indices, dtype='float32')
    mask = tf.scatter_update(mask, min_indices, ones)
    print('Mask', mask.get_shape())

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

    min_depth_vert_index_at_pixel = tf.squeeze(
        tf.cast(min_depth_vert_at_pixel[:, :, 3], dtype='int32'),
        axis=1)  # (1,)
    return min_depth_vert_index_at_pixel
