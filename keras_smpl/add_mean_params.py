
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import deepdish as dd


def add_mean_params(smpl_diff):
    mean = np.zeros((1, 87))
    mean[0, 0] = 10.0
    mean[0, 1] = 10.0
    mean[0, 2] = 0.0
    mean[0, 3] = 0.0
    mean[0, 4] = 10.0
    # mean_path = './neutral_smpl_mean_params.h5'
    # mean_vals = dd.io.load(mean_path)
    # # print(mean_vals)
    #
    # mean_pose = mean_vals['pose']
    # # Ignore the global rotation.
    # mean_pose[:3] = 0.
    # mean_shape = mean_vals['shape']
    #
    # # # This initializes the global pose to be up-right when projected
    # # mean_pose[0] = np.pi
    # #
    # mean[0, 5:] = np.hstack((mean_pose, mean_shape))
    mean = tf.constant(mean, tf.float32)
    # mean = tf.tile(mean, [batch_size, 1])
    smpl = tf.add(smpl_diff, mean)
    return smpl
