"""
https://arxiv.org/pdf/1708.02002.pdf
"""

from keras import backend as K
import tensorflow as tf


def categorical_focal_loss(gamma=2.0):

    def categorical_focal_loss_fixed(y_true, y_pred):
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # ^ this should be unecessary since y_pred is from softmax - I think its for stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # prevent Infs and NaNs by clipping
        cross_entropy = -y_true * K.log(y_pred)
        focal_loss = K.pow(1-y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss, axis=2)  # sum over classes dimension (only non zero value in sum is -log(correct class output)
        print('FOCAL LOSS', focal_loss.get_shape())
        return focal_loss

    return categorical_focal_loss_fixed


