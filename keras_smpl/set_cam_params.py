
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import deepdish as dd


def set_cam_params(smpl):
    mean = np.zeros((1, 86))
    mean[0, 0] = 32.0
    mean[0, 1] = 32.0
    mean[0, 2] = 32.0
    mean[0, 3] = 32.0
    mean = tf.constant(mean, tf.float32)
    # mean = tf.tile(mean, [batch_size, 1])
    output_smpl = tf.add(smpl, mean)
    return output_smpl
