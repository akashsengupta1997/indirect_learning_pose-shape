import os

from matplotlib import pyplot as plt
import numpy as np
import cv2
import pickle

import tensorflow as tf
from keras.models import load_model
from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project
from keras_smpl.projects_to_seg import projects_to_seg


def test(model, img_wh, img_dec_wh, image_dir, num_classes, save=False):
    img_list = []
    fnames = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            print(fname)
            image = cv2.imread(os.path.join(image_dir, fname))
            image = cv2.resize(image, (img_wh, img_wh))
            image = image[..., ::-1]
            # plt.imshow(image)
            # plt.show()
            img_list.append(image / 255.0)
            fnames.append(fname)

    img_tensor = np.array(img_list)
    output = np.reshape(model.predict(img_tensor), (len(img_list), img_dec_wh, img_dec_wh,
                                                    num_classes))
    print("orig output shape", output.shape)
    for img_num in range(len(img_list)):
        seg_labels = output[img_num, :, :, :]
        seg_img = np.argmax(seg_labels, axis=2)
        # print("labels output shape", seg_labels.shape)
        # print("seg img output shape", seg_img.shape)
        if not save:
            plt.figure(1)
            plt.clf()
            plt.subplot(331)
            plt.imshow(seg_labels[:, :, 0], cmap="gray")
            plt.subplot(332)
            plt.imshow(seg_labels[:, :, 1], cmap="gray")
            plt.subplot(333)
            plt.imshow(seg_labels[:, :, 2], cmap="gray")
            plt.subplot(334)
            plt.imshow(seg_labels[:, :, 3], cmap="gray")
            plt.subplot(335)
            plt.imshow(seg_labels[:, :, 4], cmap="gray")
            plt.subplot(336)
            plt.imshow(seg_labels[:, :, 5], cmap="gray")
            plt.subplot(337)
            plt.imshow(seg_labels[:, :, 6], cmap="gray")
            plt.figure(2)
            plt.clf()
            plt.imshow(seg_img)
            plt.figure(3)
            plt.clf()
            plt.imshow(img_list[img_num])
            plt.show()
        else:
            save_path = os.path.join(image_dir, "results64_pppups31", os.path.splitext(fnames[img_num])[0]
                                     + "_seg_img.png")
            plt.imsave(save_path, seg_img*8)


def segmentation_test(img_wh, img_dec_wh, num_classes, save=False):
    test_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images/train'
    print('Preloaded model')
    model = load_model('./test_models/ups31_0101.hdf5',
                       custom_objects={'SMPLLayer': SMPLLayer,
                                       'tf': tf,
                                       'pickle': pickle})
    test(model, img_wh, img_dec_wh, test_image_dir, num_classes, save=save)


segmentation_test(256, 64, 32, save=False)

