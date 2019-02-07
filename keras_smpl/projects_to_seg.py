# TODO write function that takes verts and camera params as input and goes to bodypart seg
# can then implement this as Keras lambda layer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
from keras import backend as K


def projects_to_seg(projects):
    """

    :param projects:
    :return:
    """
    part_indices_path = "./keras_smpl/part_vertices.pkl"
    img_wh = 64

    with open(part_indices_path, 'rb') as f:
        part_indices = pickle.load(f)

    i = tf.range(0, 64)
    j = tf.range(0, 64)

    t1, t2 = tf.meshgrid(i, j)
    grid = tf.cast(tf.stack([t1, t2], axis=2), dtype='float32')
    reshaped_grid = tf.reshape(grid, [-1, 2])

    segs = []
    for part in range(len(part_indices)):
        indices = part_indices[part]
        num_indices = len(indices)
        part_projects = tf.gather(projects, indices, axis=1)  # N x num_indices x 2
        part_projects = tf.tile(tf.expand_dims(part_projects, axis=1),
                                [1, img_wh*img_wh, 1, 1])  # N x 4096 x num_indices x 2
        expanded_grid = tf.tile(tf.expand_dims(reshaped_grid, axis=1),
                                [1, num_indices, 1])  # 4096 x num_indices x 2

        diff = tf.subtract(part_projects, expanded_grid)  # N x 4096 x num_indices x 2
        norm = tf.norm(diff, axis=3)  # N x 4096 x num_indices
        exp = tf.exp(tf.negative(norm))  # N x 4096 x num_indices
        scores = tf.reduce_max(exp, axis=2)  # N x 4096
        seg = tf.reshape(scores, [-1, img_wh, img_wh])  # N x 64 x 64
        segs.append(seg)

    stacked_segs = tf.stack(segs, axis=3)  # N x 64 x 64 x 31
    silhouettes = tf.subtract(1.0,
                              tf.clip_by_value(tf.reduce_sum(stacked_segs, axis=3),
                                               clip_value_min=0,
                                               clip_value_max=1))  # N x 64 x 64
    # segs.insert(0, silhouettes)
    output_segs = tf.concat([tf.expand_dims(silhouettes, axis=3), stacked_segs],
                            axis=3)  # N x 64 x 64 x 32
    output_segs = K.reverse(output_segs, axes=1)
    print(output_segs.shape)
    return output_segs

    # scores_hand = []
    # scores_torso = []
    # for i in range(img_wh):
    #     for j in range(img_wh):
    #         loc = tf.expand_dims(tf.constant([i, j], dtype='float32'), axis=0)
    #
    #         diff_hand = tf.subtract(loc, R_hand_projects)
    #         norm_hand = tf.norm(diff_hand, axis=2)
    #         exp_hand = tf.exp(tf.negative(norm_hand))
    #         score_hand = tf.reduce_max(exp_hand, axis=1)
    #         scores_hand.append(score_hand)
    #
    #         diff_torso = tf.subtract(loc, R_torso_projects)
    #         norm_torso = tf.norm(diff_torso, axis=2)
    #         exp_torso = tf.exp(tf.negative(norm_torso))
    #         score_torso = tf.reduce_max(exp_torso, axis=1)
    #         scores_torso.append(score_torso)
    #
    # R_hand_seg = tf.transpose(tf.reshape(tf.stack(scores_hand), [img_wh, img_wh, -1]))
    # R_torso_seg = tf.transpose(tf.reshape(tf.stack(scores_torso), [img_wh, img_wh, -1]))
    # print(R_hand_seg.shape)
    # print(R_torso_seg.shape)
    # return(K.eval(tf.stack([R_hand_seg, R_torso_seg], axis=3)))

    # return(K.eval(seg))


