
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import deepdish as dd

from keras import backend as K


def set_cam_params(smpl):
    mean = np.zeros((1, 86))
    mean[0, 0] = 64.0
    mean[0, 1] = 64.0
    mean[0, 2] = 64.0
    mean[0, 3] = 80.0
    mean = tf.constant(mean, tf.float32)
    mean = tf.tile(mean, [K.shape(smpl)[0], 1])
    output_smpl = tf.add(smpl, mean)
    return output_smpl


def load_mean_set_cam_params(smpl):
    mean = np.zeros((1, 86))
    mean[0, 0] = 64.0
    mean[0, 1] = 64.0
    mean[0, 2] = 64.0
    mean[0, 3] = 80.0

    mean_path = './neutral_smpl_mean_params.h5'
    mean_smpl = dd.io.load(mean_path)
    mean_pose = mean_smpl['pose']
    # Ignore the global rotation.
    mean_pose[:3] = 0.
    mean_shape = mean_smpl['shape']
    mean[0, 4:] = np.hstack((mean_pose, mean_shape))

    mean = tf.constant(mean, dtype=tf.float32)
    mean = tf.tile(mean, [K.shape(smpl)[0], 1])
    output_smpl = tf.add(smpl, mean)
    return output_smpl
