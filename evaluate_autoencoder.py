import os

import numpy as np
import deepdish as dd
import cv2

import tensorflow as tf
from keras.models import load_model

from model import build_full_model_for_predict


def load_input_seg(eval_dir, fname, input_wh, num_classes):
    input_seg = cv2.imread(os.path.join(eval_dir, fname), 0)
    input_seg = cv2.resize(input_seg, (input_wh, input_wh),
                           interpolation=cv2.INTER_NEAREST)
    input_seg = np.expand_dims(input_seg, axis=-1)
    input_seg = input_seg * (1.0 / (num_classes - 1))
    input_seg = np.expand_dims(input_seg, axis=0)  # need 4D input (add batch dimension)
    return input_seg


def compute_intersection_and_union(ground_truth, predict, num_classes):
    """
    Compute number of intersections and unions between ground truth label and predicted label
    (for 1 training example).
    :param ground_truth:
    :param predict:
    :param num_classes:
    :return: num_intersections, num_unions
    """
    num_intersections_per_class = []
    num_unions_per_class = []
    for class_num in range(1, num_classes):  # not including background class
        ground_truth_binary = np.zeros(ground_truth.shape)
        predict_binary = np.zeros(predict.shape)
        ground_truth_binary[ground_truth == class_num] = 1
        predict_binary[predict == class_num] = 1

        intersection = np.logical_and(ground_truth_binary, predict_binary)
        union = np.logical_or(ground_truth_binary, predict_binary)
        num_intersections = float(np.sum(intersection))
        num_unions = float(np.sum(union))
        num_intersections_per_class.append(num_intersections)
        num_unions_per_class.append(num_unions)
        # print(num_intersections, num_unions)

    return np.array(num_intersections_per_class), np.array(num_unions_per_class)


def count_correct_predicts(ground_truth, predict):
    """
    Counts number of correct predicts in 1 training example.
    :param ground_truth:
    :param predict:
    :return: pixel accuracy
    """
    correct_predicts = np.equal(ground_truth, predict)
    num_correct_predicts = float(np.sum(correct_predicts))

    return num_correct_predicts


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

    verts_model, projects_model, segs_model = build_full_model_for_predict(smpl_model,
                                                                           output_wh,
                                                                           "./neutral_smpl_with_cocoplus_reg.pkl")
    total_intersects_per_class = np.zeros(num_classes - 1)
    total_unions_per_class = np.zeros(num_classes - 1)
    total_correct_predicts = 0
    num_eval_images = len([fname for fname in sorted(os.listdir(eval_dir)) if
                           fname.endswith(".png")])
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

            num_intersects_per_class, num_unions_per_class = compute_intersection_and_union(
                ground_truth_seg_map,
                predicted_seg_map,
                num_classes)
            print("I/Us for image", num_intersects_per_class, num_unions_per_class)
            total_intersects_per_class += num_intersects_per_class
            total_unions_per_class += num_unions_per_class
            print("Total I/Us", total_intersects_per_class, total_unions_per_class)

            num_correct_predicts = count_correct_predicts(ground_truth_seg_map,
                                                          predicted_seg_map)
            total_correct_predicts += num_correct_predicts
            print("Correct predicts for image", num_correct_predicts)
            print("Total correct predicts", total_correct_predicts)

    ious = np.divide(total_intersects_per_class, total_unions_per_class)
    mean_iou = np.mean(ious)
    print("Mean IOU", mean_iou)
    accuracy = total_correct_predicts / (output_wh * output_wh * num_eval_images)
    print("Overall accuracy over dataset", accuracy)


def evaluate_autoencoder_from_enet_segs_iou_and_acc(enet_segs_dir, gt_segs_dir, input_wh, output_wh,
                                                    num_classes, model_fname):
    smpl_model = load_model(os.path.join("./autoencoder_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model_for_predict(smpl_model,
                                                                            output_wh,
                                                                            "./neutral_smpl_with_cocoplus_reg.pkl")
    total_intersects_per_class = np.zeros(num_classes - 1)
    total_unions_per_class = np.zeros(num_classes - 1)
    total_correct_predicts = 0
    num_eval_images = len([fname for fname in sorted(os.listdir(enet_segs_dir)) if
                           fname.endswith(".png")])
    for fname in sorted(os.listdir(enet_segs_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_seg_map = load_input_seg(enet_segs_dir, fname, input_wh, num_classes)  # (1, input_wh, input_wh, 1)
            segs = segs_model.predict(input_seg_map)  # (1, output_wh, output_wh, 32)
            predicted_seg_map = np.argmax(segs, axis=-1)
            predicted_seg_map = predicted_seg_map[0]  # (output_wh, output_wh)
            ground_truth_seg_map = cv2.resize(cv2.imread(os.path.join(gt_segs_dir,
                                                                      fname[:5] + "_ann.png"),
                                                         0),
                                              (output_wh, output_wh),
                                              interpolation=cv2.INTER_NEAREST)  # (output_wh, output_wh)

            num_intersects_per_class, num_unions_per_class = compute_intersection_and_union(
                ground_truth_seg_map,
                predicted_seg_map,
                num_classes)
            print("I/Us for image", num_intersects_per_class, num_unions_per_class)
            total_intersects_per_class += num_intersects_per_class
            total_unions_per_class += num_unions_per_class
            print("Total I/Us", total_intersects_per_class, total_unions_per_class)

            num_correct_predicts = count_correct_predicts(ground_truth_seg_map,
                                                          predicted_seg_map)
            total_correct_predicts += num_correct_predicts
            print("Correct predicts for image", num_correct_predicts)
            print("Total correct predicts", total_correct_predicts)

            # plt.figure(1)
            # plt.subplot(311)
            # plt.imshow(input_seg_map[0, :, :, 0])
            # plt.subplot(312)
            # plt.imshow(predicted_seg_map)
            # plt.subplot(313)
            # plt.imshow(ground_truth_seg_map)
            # plt.show()

    ious = np.divide(total_intersects_per_class, total_unions_per_class)
    mean_iou = np.mean(ious)
    print("Mean IOU", mean_iou)
    accuracy = total_correct_predicts / (output_wh * output_wh * num_eval_images)
    print("Overall accuracy over dataset", accuracy)

# evaluate_autoencoder_iou_and_acc("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val",
#                                  256,
#                                  48,
#                                  32,
#                                  'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5')

evaluate_autoencoder_from_enet_segs_iou_and_acc("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot_enet_segs",
                                                "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val",
                                                256,
                                                48,
                                                32,
                                                'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5')