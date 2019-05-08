"""
Basically a copy of projects_to_seg that only returns silhouette - very hacky.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
from keras import backend as K


def projects_to_silhouette(projects_with_depth, img_wh):
    """

    :param projects:
    :return:
    """
    projects = projects_with_depth[:, :, :2]

    i = tf.range(0, img_wh)
    j = tf.range(0, img_wh)

    t1, t2 = tf.meshgrid(i, j)
    grid = tf.cast(tf.stack([t1, t2], axis=2), dtype='float32')  # img_wh x img_wh x 2
    reshaped_grid = tf.reshape(grid, [-1, 2])  # img_wh^2 x 2

    projects = tf.tile(tf.expand_dims(projects, axis=1),
                            [1, img_wh * img_wh, 1, 1])  # N x img_wh^2 x 6890 x 2

    expanded_grid = tf.tile(tf.expand_dims(reshaped_grid, axis=1),
                            [1, 6890, 1])  # img_wh^2 x 6890 x 2

    diff = tf.subtract(projects, expanded_grid)  # N x img_wh^2 x 6890 x 2
    norm = tf.norm(diff, axis=3, name='big_norm1')  # N x img_wh^2 x 6890
    exp = tf.exp(tf.negative(norm)/1.2)  # N x img_wh^2 x 6890  # 1.2 = variance of gaussian
    scores = tf.reduce_max(exp, axis=2)  # N x img_wh^2
    silhouettes = tf.reshape(scores, [-1, img_wh, img_wh])  # N x img_wh x img_wh
    backgrounds = tf.subtract(1.0, silhouettes)  # N x img_wh x img_wh
    output = tf.stack([backgrounds, silhouettes], axis=3)  # N x img_wh x img_wh x 2
    output = tf.reverse(output, axis=[1])  # Flip image vertically

    return output



