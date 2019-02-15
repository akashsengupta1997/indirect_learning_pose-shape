import tensorflow as tf
import numpy as np
import deepdish as dd

from keras import backend as K


def load_mean_param(num_smpl_params):
    mean = np.zeros((1, num_smpl_params))
    mean_path = './neutral_smpl_mean_params.h5'
    mean_vals = dd.io.load(mean_path)

    mean_pose = mean_vals['pose']
    # Ignore the global rotation.
    mean_pose[:3] = 0.
    mean_shape = mean_vals['shape']

    # # This initializes the global pose to be up-right when projected
    # mean_pose[0] = np.pi

    mean[0, :] = np.hstack((mean_pose, mean_shape))
    mean = tf.constant(mean, dtype=tf.float32)
    # init_mean = tf.tile(mean, [self.batch_size, 1])
    # return init_mean
    return mean


def concat_mean_param(img_features):
    mean_path = './neutral_smpl_mean_params.h5'
    mean_vals = dd.io.load(mean_path)

    mean_pose = mean_vals['pose']
    # Ignore the global rotation.
    mean_pose[:3] = 0.
    mean_shape = mean_vals['shape']

    # # This initializes the global pose to be up-right when projected
    # mean_pose[0] = np.pi

    mean = np.expand_dims(np.hstack((mean_pose, mean_shape)), axis=0)
    mean = tf.constant(mean, dtype='float32')
    # print(mean.shape)
    mean = tf.tile(mean, [K.shape(img_features)[0], 1])

    state = tf.concat([img_features, mean], axis=1)
    return [state, mean]

# concat_mean_param(None)