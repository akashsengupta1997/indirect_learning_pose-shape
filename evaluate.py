import os

import numpy as np
import deepdish as dd
import cv2

import tensorflow as tf
from keras.models import load_model

from model import build_full_model_for_predict


def load_input_img(image_dir, fname, input_wh):
    input_img = cv2.imread(os.path.join(image_dir, fname))
    input_img = cv2.resize(input_img, (input_wh, input_wh))
    input_img = input_img[..., ::-1]
    input_img = input_img * (1.0 / 255)
    input_img = np.expand_dims(input_img, axis=0)  # need 4D input (add batch dimension)
    return input_img


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


def evaluate_iou_and_acc(eval_image_dir, eval_mask_dir, input_wh, output_wh, num_classes,
                         model_fname):
    """
    Evaluate given indirect learn model (model_fname) over test dataset specified by eval_dir.
    Prints IOU and pixel-wise accuracies.
    :param eval_dir: file path to evaluation dataset
    :param input_wh:
    :param output_wh:
    :param num_classes:
    :param model_fname: file path to weights of model to be evaluated
    """
    smpl_model = load_model(os.path.join("./full_network_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model_for_predict(smpl_model,
                                                                           output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl")

    total_intersects_per_class = np.zeros(31)
    total_unions_per_class = np.zeros(31)
    total_correct_predicts = 0
    num_eval_images = len([fname for fname in sorted(os.listdir(eval_image_dir)) if
                           fname.endswith(".png")])
    print ("Evaluation set size:", num_eval_images)
    for fname in sorted(os.listdir(eval_image_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_img = load_input_img(eval_image_dir, fname, input_wh)
            segs = segs_model.predict(input_img)  # (1, output_wh, output_wh, 32)
            predicted_seg_map = np.argmax(segs, axis=-1)
            predicted_seg_map = predicted_seg_map[0]  # (output_wh, output_wh)
            ground_truth_seg_map = cv2.resize(cv2.imread(os.path.join(eval_mask_dir,
                                                                      fname[:5]+"_ann.png"),
                                                         0),
                                              (output_wh, output_wh),
                                              interpolation=cv2.INTER_NEAREST)  # (output_wh, output_wh)

            num_intersects_per_class, num_unions_per_class = compute_intersection_and_union(ground_truth_seg_map,
                                                                                            predicted_seg_map,
                                                                                            num_classes)
            print("I/Us for image", num_intersects_per_class, num_unions_per_class)
            total_intersects_per_class += num_intersects_per_class
            total_unions_per_class += num_unions_per_class
            print("Total I/Us", total_intersects_per_class, total_unions_per_class)

            # plt.figure(1)
            # plt.subplot(211)
            # plt.imshow(predicted_seg_map)
            # plt.subplot(212)
            # plt.imshow(ground_truth_seg_map)
            # plt.show()

            num_correct_predicts = count_correct_predicts(ground_truth_seg_map,
                                                          predicted_seg_map)
            total_correct_predicts += num_correct_predicts
            print("Correct predicts for image", num_correct_predicts)
            print("Total correct predicts", total_correct_predicts)

    ious = np.divide(total_intersects_per_class, total_unions_per_class)
    mean_iou = np.mean(ious)
    print("Mean IOU", mean_iou)
    accuracy = total_correct_predicts/(output_wh*output_wh*num_eval_images)
    print("Overall accuracy over dataset", accuracy)


evaluate_iou_and_acc("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_images/val",
                     "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val",
                     256,
                     48,
                     32,
                     'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5')
