import os

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Lambda

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import orthographic_project
from keras_smpl.compute_mask import compute_mask
from keras_smpl.projects_to_seg import projects_to_seg


def load_input_seg(eval_dir, fname, input_wh, num_classes):
    input_seg = cv2.imread(os.path.join(eval_dir, fname), 0)
    input_seg = cv2.resize(input_seg, (input_wh, input_wh),
                           interpolation=cv2.INTER_NEAREST)
    input_seg = np.expand_dims(input_seg, axis=-1)
    input_seg = input_seg * (1.0 / (num_classes - 1))
    input_seg = np.expand_dims(input_seg, axis=0)  # need 4D input (add batch dimension)
    return input_seg


def build_full_model(smpl_model, output_wh, smpl_path, batch_size=1):
    inp = smpl_model.input
    smpl = smpl_model.output
    verts = SMPLLayer(smpl_path, batch_size=batch_size)(smpl)
    projects_with_depth = Lambda(orthographic_project,
                                 arguments={'vertex_sampling': None},
                                 name='project')([verts, smpl])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': None},
                  name='segment')([projects_with_depth, masks])

    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)
    segs_model = Model(inputs=inp, outputs=segs)

    return verts_model, projects_model, segs_model


def compute_mean_iou(ground_truth, predict, num_classes):
    """
    Compute IoU averaged over all classes between ground truth label and predicted label
    :param ground_truth:
    :param predict:
    :param num_classes:
    :return: mean IOU over all classes for 1 training example
    """
    class_ious = []
    for class_num in range(1, num_classes):  # not including background class
        ground_truth_binary = np.zeros(ground_truth.shape)
        predict_binary = np.zeros(predict.shape)
        ground_truth_binary[ground_truth == class_num] = 1
        predict_binary[predict == class_num] = 1

        intersection = np.logical_and(ground_truth_binary, predict_binary)
        union = np.logical_or(ground_truth_binary, predict_binary)
        if np.sum(union) != 0:  # Don't include if no occurences of class in image
            iou_score = float(np.sum(intersection)) / np.sum(union)
            class_ious.append(iou_score)

    return np.mean(class_ious)


def compute_pixel_accuracy(ground_truth, predict):
    """
    Compute pixel-wise class accuracy for 1 training example.
    :param ground_truth:
    :param predict:
    :return: pixel accuracy
    """
    correct_predicts = np.equal(ground_truth, predict)
    num_correct_predicts = np.sum(correct_predicts)
    accuracy = num_correct_predicts/float(ground_truth.size)

    return accuracy


def evaluate_autoencoder_iou_and_acc(eval_dir, input_wh, output_wh, num_classes, model_fname):
    """
    Evaluate given autoencoder model (model_fname) over test dataset specified by eval_dir. Prints IOU and
    pixel-wise accuracies.
    :param eval_dir: file path to evaluation dataset
    :param input_wh:
    :param output_wh:
    :param num_classes:
    :param model_fname: file path to weights of model to be evaluated
    """
    smpl_model = load_model(os.path.join("./autoencoder_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model(smpl_model,
                                                               output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl")

    ious = []
    accuracies = []
    for fname in sorted(os.listdir(eval_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_seg_map = load_input_seg(eval_dir, fname, input_wh, num_classes)  # (1, input_wh, input_wh, 1)
            segs = segs_model.predict(input_seg_map)  # (1, output_wh, output_wh, 32)
            predicted_seg_map = np.argmax(segs, axis=-1)
            predicted_seg_map = predicted_seg_map[0]  # (output_wh, output_wh)
            ground_truth_seg_map = cv2.resize(cv2.imread(os.path.join(eval_dir, fname), 0),
                                              (output_wh, output_wh),
                                              interpolation=cv2.INTER_NEAREST)  # (output_wh, output_wh)

            iou = compute_mean_iou(ground_truth_seg_map, predicted_seg_map, num_classes)
            print("IOU", iou)
            accuracy = compute_pixel_accuracy(ground_truth_seg_map, predicted_seg_map)
            print("Accuracy", accuracy)
            ious.append(iou)
            accuracies.append(accuracy)

    mean_iou_over_dataset = np.mean(ious)
    mean_acc_over_dataset = np.mean(accuracies)
    print("Mean IOU over dataset", mean_iou_over_dataset)
    print("Mean accuracy over dataset", mean_acc_over_dataset)


evaluate_autoencoder_iou_and_acc("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val",
                                 256,
                                 48,
                                 32,
                                 'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5')
