import os

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2
import pickle

import tensorflow as tf
from keras.models import load_model


def load_input_seg(eval_dir, fname, input_wh, num_classes):
    input_seg = cv2.imread(os.path.join(eval_dir, fname), 0)
    input_seg = cv2.resize(input_seg, (input_wh, input_wh),
                           interpolation=cv2.INTER_NEAREST)
    input_seg = np.expand_dims(input_seg, axis=-1)
    input_seg = input_seg * (1.0 / (num_classes - 1))
    input_seg = np.expand_dims(input_seg, axis=0)  # need 4D input (add batch dimension)
    return input_seg


def load_gt_pose(fname):
    gt_smpl_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/gt_smpl"
    with open(os.path.join(gt_smpl_dir, fname), 'rb') as f:
        data = pickle.load(f)
        gt_pose = data['pose']
        gt_pose_no_glob_rot = gt_pose[3:]

    return gt_pose_no_glob_rot


def evaluate_pose_param_mse(eval_dir, input_wh, num_classes, model_fname):
    smpl_model = load_model(os.path.join("./autoencoder_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    squared_errors = []
    for fname in sorted(os.listdir(eval_dir)):
        if fname.endswith(".png"):
            print(fname)
            fnumber = fname[:5]
            gt_smpl_fname = fnumber + "_body.pkl"
            gt_pose_no_glob_rot = load_gt_pose(gt_smpl_fname)

            input_seg_map = load_input_seg(eval_dir, fname, input_wh, num_classes)
            pred_smpl = smpl_model.predict(input_seg_map)[0]
            pred_pose_no_glob_rot = pred_smpl[7:76]
            error = np.square(gt_pose_no_glob_rot - pred_pose_no_glob_rot)
            squared_errors.append(error)

            # print("error", error)
            # print("history", squared_errors)
            # print("mse for example", np.mean(error))

    squared_errors = np.concatenate(squared_errors)
    print(squared_errors.shape)
    print("MSE pose params", np.mean(squared_errors))


evaluate_pose_param_mse("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val",
                        256,
                        'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5')

# evaluate_pose_param_mse("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot_enet_segs",
#                         256,
#                         'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5')