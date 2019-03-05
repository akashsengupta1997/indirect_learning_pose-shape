
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import deepdish as dd

from keras import backend as K


def set_cam_params(smpl, img_wh):
    cam_params = np.zeros((1, 86))
    # cam_params[0, 0] = 48.0  # scaling in x direction
    # cam_params[0, 1] = 48.0  # scaling in y direction
    # cam_params[0, 2] = 48.0  # translation in x direction
    # cam_params[0, 3] = 60.0  # translation in y direction
    cam_params[0, 0] = img_wh/2.0  # scaling in x direction
    cam_params[0, 1] = img_wh/2.0   # scaling in y direction
    cam_params[0, 2] = img_wh/2.0   # translation in x direction
    cam_params[0, 3] = img_wh/1.6   # translation in y direction
    cam_params = tf.constant(cam_params, tf.float32)
    cam_params = tf.tile(cam_params, [K.shape(smpl)[0], 1])
    output_smpl = tf.add(smpl, cam_params)
    return output_smpl


def load_mean_set_cam_params(smpl, img_wh):
    mean = np.zeros((1, 86))
    # mean[0, 0] = 48.0
    # mean[0, 1] = 48.0
    # mean[0, 2] = 48.0
    # mean[0, 3] = 60.0

    mean[0, 0] = img_wh/2.0
    mean[0, 1] = img_wh/2.0
    mean[0, 2] = img_wh/2.0
    mean[0, 3] = img_wh/1.6

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
