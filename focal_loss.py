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

            # weights = np.ones(32)
            # weights[0] = 0.3
            # weights[1] = 2.0
            # weights[2] = 2.0
            # weights[3] = 2.0
            # weights[4] = 2.0
            #
            # weights[10] = 2.0
            # weights[12] = 2.0
            #
            # weights[14] = 2.0
            # weights[15] = 2.0
            # weights[16] = 2.0
            # weights[17] = 2.0
            #
            # weights[23] = 2.0
            # weights[25] = 2.0

            weights = [1.03324042e-04, 3.02493745e-02, 1.30634496e-01, 2.98523010e-02,
                       3.16610807e-02, 3.16602506e-02, 2.03280074e-02, 8.58922762e-03,
                       6.56260681e-03, 9.47177711e-03, 1.57712035e-02, 1.34693966e-02,
                       4.45866666e-02, 3.52281362e-02, 3.23108385e-02, 1.31530909e-01,
                       3.13854163e-02, 3.26127601e-02, 3.36627904e-02, 1.92015468e-02,
                       8.54062987e-03, 6.74304680e-03, 9.33962334e-03, 1.55358396e-02,
                       1.35297609e-02, 4.52940730e-02, 3.58567991e-02, 2.79993804e-02,
                       3.02493745e-02, 4.28206546e-02, 4.34602058e-02, 3.17585020e-02]

            weights = tf.constant(weights, dtype='float32')
            cross_entropy = tf.multiply(cross_entropy, weights)
            print("WEIGHTED LOSS")

        focal_loss = K.pow(1-y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss, axis=2)  # sum over classes dimension (only non zero value in sum is -log(correct class output)
        print('FOCAL LOSS', focal_loss.get_shape())
        return focal_loss

    return categorical_focal_loss_fixed

