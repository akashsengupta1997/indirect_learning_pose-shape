# TODO write function that takes verts and camera params as input and goes to bodypart seg
# can then implement this as Keras lambda layer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle


def projects_to_seg(projects):
    """

    :param projects:
    :return:
    """
    part_indices_path = "part_vertices.pkl"
    img_wh = 64

    with open(part_indices_path, 'rb') as f:
        part_indices = pickle.load(f)

        R_hand_seg = tf.zeros([img_wh, img_wh])
        R_hand_indices = part_indices[0]
        R_hand_projects = tf.gather(projects, R_hand_indices, axis=1)
        print(R_hand_projects.shape)

        for i in range(img_wh):
            for j in range(img_wh):
                loc = tf.expand_dims(tf.constant([i, j], dtype=tf.float32), axis=1)
                print(loc.shape)
                diff = tf.subtract(loc)

                # score = tf.exp()
                # R_hand_seg[i, j] =




