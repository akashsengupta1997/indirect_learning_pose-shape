"""
https://arxiv.org/pdf/1708.02002.pdf
"""

from keras import backend as K
import tensorflow as tf
import numpy as np


def categorical_focal_loss(gamma=2.0, weight_classes=False):

    def categorical_focal_loss_fixed(y_true, y_pred):
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # ^ this should be unecessary since y_pred is from softmax - I think its for stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # prevent Infs and NaNs by clipping
        cross_entropy = -y_true * K.log(y_pred)  # (N, img_wh^2, 32)

        if weight_classes:

            weights = np.ones(32)
            weights[0] = 0.3
            weights[1] = 2.0
            weights[2] = 2.0
            weights[3] = 2.0
            weights[4] = 2.0

            weights[10] = 2.0
            weights[12] = 2.0

            weights[14] = 2.0
            weights[15] = 2.0
            weights[16] = 2.0
            weights[17] = 2.0

            weights[23] = 2.0
            weights[25] = 2.0
            weights = tf.constant(weights, dtype='float32')
            cross_entropy = tf.multiply(cross_entropy, weights)
            print("WEIGHTED LOSS")

        focal_loss = K.pow(1-y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss, axis=2)  # sum over classes dimension (only non zero value in sum is -log(correct class output)
        print('FOCAL LOSS', focal_loss.get_shape())
        return focal_loss

    return categorical_focal_loss_fixed

