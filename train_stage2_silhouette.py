"""
Script for second stage of training - using silhouette labels instead of body part labels.
"""

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
import deepdish as dd
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Lambda, Reshape, Activation

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project2
from keras_smpl.projects_to_silhouette import projects_to_silhouette
from focal_loss import binary_focal_loss

from renderer import SMPLRenderer


def binary_classlab(labels, num_classes=2):
    """
    Function to convert HxWx1 labels image to HxWxC one hot encoded matrix.
    :param labels: HxWx1 labels image
    :param num_classes: number of segmentation classes
    :return: HxWxC one hot encoded matrix.
    """
    x = np.zeros((labels.shape[0], labels.shape[1], num_classes))
    # print('IN CLASSLAB', labels.shape)
    for pixel_class in range(num_classes):
        indexes = list(zip(*np.where(labels == pixel_class)))
        for index in indexes:
            x[index[0], index[1], pixel_class] = 1.0
    return x


def generate_data(image_generator, mask_generator, n, num_classes=2):
    """
    Generate 1 batch of training data (batch_size = n) from given keras generators (made using
    flow_from_directory method).
    Pre-processes mask generator output to go from H x W grayscale image to H x W x num_classes
    labels.
    :param image_generator: input image generator
    :param mask_generator: output segmentation generator
    :param n: batch size
    :param num_classes: numberof segmentation classes
    :return: np arrays of 1 batch of input images and 1 batch of output labels.
    """
    images = []
    labels = []
    i = 0
    while i < n:
        x = image_generator.next()
        y = mask_generator.next()
        j = 0
        while j < x.shape[0]:
            images.append(x[j, :, :, :])
            labels.append(binary_classlab(y[j, :, :, :].astype(np.uint8), num_classes))
            j = j + 1
            i = i + 1
            if i >= n:
                break
    return np.array(images), np.array(labels)


def build_full_model_from_saved_model(smpl_model, output_wh, smpl_path, batch_size,
                                      num_classes):
    inp = smpl_model.input
    smpl = smpl_model.output
    verts = SMPLLayer(smpl_path, batch_size=batch_size)(smpl)
    projects_with_depth = Lambda(orthographic_project2,
                                 arguments={'vertex_sampling': None},
                                 name='project')([verts, smpl])
    silhouettes = Lambda(projects_to_silhouette,
                         arguments={'img_wh': output_wh},
                         name='segment')(projects_with_depth)
    silhouettes = Reshape((output_wh * output_wh, num_classes), name="final_reshape")(silhouettes)
    silhouettes = Activation('softmax', name="final_softmax")(silhouettes)

    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)
    silhouettes_model = Model(inputs=inp, outputs=silhouettes)

    print(silhouettes_model.summary())

    return verts_model, projects_model, silhouettes_model


def train(resume_from, input_wh, output_wh, save_model=False):
    batch_size = 4
    train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/upi-s1h/trial_images/"
    train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/upi-s1h/trial_masks/"
    # train_image_dir = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii_padded/images"
    # train_label_dir = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii_padded/masks"
    monitor_dir = "./second_stage_monitor/monitor_train_images"
    # TODO create validation directory
    num_classes = 2
    num_train_images = 13030

    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    # assert os.path.isdir(val_image_dir), 'Invalid validation image directory'
    # assert os.path.isdir(val_label_dir), 'Invalid validation label directory'

    val_image_data_gen_args = dict(
        rescale=(1/255.0),
        fill_mode='nearest')

    val_mask_data_gen_args = dict(
        rescale=(1 / 255.0),
        fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

    # Provide the same seed to flow methods for train generators
    seed = 1
    train_image_generator = train_image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(input_wh, input_wh),
        class_mode=None,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        target_size=(output_wh, output_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    print('Generators loaded.')

    # # For testing data loading
    # x = train_image_generator.next()
    # y = train_mask_generator.next()
    # print('x shape out of training generator', x.shape)  # should = (batch_size, img_hw, img_hw, 3)
    # print('y shape out of training generator', y.shape)  # should = (batch_size, dec_hw, dec_hw, 1)
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(x[0, :, :, :])
    # plt.subplot(222)
    # plt.imshow(y[0, :, :, 0])
    # y_post = binary_classlab(y[0], num_classes)
    # plt.subplot(223)
    # plt.imshow(y_post[:, :, 0])
    # plt.subplot(224)
    # plt.imshow(y_post[:, :, 1])
    # plt.show()

    print("Resuming model from ", resume_from)
    smpl_model = load_model(os.path.join("./full_network_weights", resume_from),
                            custom_objects={'dd': dd,
                                            'tf': tf})

    verts_model, projects_model, silhouettes_model = \
        build_full_model_from_saved_model(smpl_model,
                                          output_wh,
                                          "./neutral_smpl_with_cocoplus_reg.pkl",
                                          batch_size,
                                          num_classes)
    print("Model loaded.")

    adam_optimiser = Adam(lr=0.0001)
    silhouettes_model.compile(optimizer=adam_optimiser,
                       loss=binary_focal_loss(gamma=2.0, weight_classes=True),
                       metrics=['accuracy'])

    print("Model compiled.")

    for trial in range(4000):
        print("Fitting", trial)

        def train_data_gen():
            while True:
                train_data, train_labels = generate_data(train_image_generator,
                                                         train_mask_generator,
                                                         batch_size,
                                                         num_classes)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, output_wh * output_wh,
                                                    num_classes))
                yield (train_data, reshaped_train_labels)

        history = silhouettes_model.fit_generator(train_data_gen(),
                                                  steps_per_epoch=int((num_train_images/batch_size)/5),
                                                  nb_epoch=1,
                                                  verbose=1)

        renderer = SMPLRenderer()

        if trial % 20 == 0:
            inputs = []
            for fname in sorted(os.listdir(monitor_dir)):
                if fname.endswith(".png"):
                    input_image = cv2.imread(os.path.join(monitor_dir, fname), 1)
                    input_image = cv2.resize(input_image,
                                             (input_wh, input_wh),
                                             interpolation=cv2.INTER_NEAREST)
                    input_image = input_image[..., ::-1]
                    input_image = input_image * (1.0/255)
                    inputs.append(input_image)

            input_images_array = np.array(inputs)
            input_images_array1 = input_images_array[:batch_size, :, :, :]
            input_images_array2 = input_images_array[batch_size:, :, :, :]

            smpls1 = smpl_model.predict(input_images_array1)
            verts1 = verts_model.predict(input_images_array1)
            projects1 = projects_model.predict(input_images_array1)
            segs1 = np.reshape(silhouettes_model.predict(input_images_array1),
                               [-1, output_wh, output_wh, num_classes])

            smpls2 = smpl_model.predict(input_images_array2)
            verts2 = verts_model.predict(input_images_array2)
            projects2 = projects_model.predict(input_images_array2)
            segs2 = np.reshape(silhouettes_model.predict(input_images_array2),
                               [-1, output_wh, output_wh, num_classes])

            smpls = np.concatenate((smpls1, smpls2), axis=0)
            verts = np.concatenate((verts1, verts2), axis=0)
            projects = np.concatenate((projects1, projects2), axis=0)
            segs = np.concatenate((segs1, segs2), axis=0)

            seg_maps = np.argmax(segs, axis=-1)

            print(smpls[0])
            for i in range(smpls.shape[0]):
                plt.figure(1)
                plt.clf()
                plt.imshow(seg_maps[i])
                plt.savefig("./second_stage_monitor/seg_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(2)
                plt.clf()
                plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./second_stage_monitor/verts_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(3)
                rend_img = renderer(verts=verts[i], render_seg=False)
                plt.imshow(rend_img)
                plt.savefig("./second_stage_monitor/rend_" + str(trial) + "_" + str(i) + ".png")

                if trial == 0:
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(input_images_array[i, :, :, :])
                    plt.savefig("./second_stage_monitor/image_" + str(i) + ".png")

            if save_model:
                save_fname = "second_stage_" + str(trial) + "_" + resume_from
                smpl_model.save(os.path.join('./test_models', save_fname))
                print('SAVE NAME', save_fname)

    print("Finished")


train("up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5",
      256,
      96,
      save_model=True)

