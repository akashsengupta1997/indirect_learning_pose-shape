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
from keras_smpl.projection import orthographic_project2
from keras_smpl.projects_to_silhouette import projects_to_silhouette
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.compute_mask import compute_mask
from focal_loss import categorical_focal_loss

from renderer import SMPLRenderer


def classlab(labels, num_classes):
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


def generate_data(image_generator, mask_generator, n, num_classes):
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
            labels.append(classlab(y[j, :, :, :].astype(np.uint8), num_classes))
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
                         name='segment_silh')(projects_with_depth)
    silhouettes = Reshape((output_wh * output_wh, num_classes), name="final_reshape_silh")(silhouettes)
    silhouettes = Activation('softmax', name="final_softmax_silh")(silhouettes)

    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': None},
                  name='segment_bodyparts')([projects_with_depth, masks])
    segs = Reshape((output_wh * output_wh, num_classes), name="final_reshape_segs")(segs)
    segs = Activation('softmax', name="final_softmax_segs")(segs)

    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)
    silhouettes_model = Model(inputs=inp, outputs=silhouettes)
    segs_model = Model(inputs=inp, outputs=segs)

    print(silhouettes_model.summary())

    return verts_model, projects_model, silhouettes_model, segs_model


def train(resume_from, input_wh, segs_output_wh, silhs_output_wh, save_model=False, weight_segs_classes=True):
    batch_size = 4
    # train_image_dir_segs = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images/"
    # train_label_dir_segs = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks/"
    # train_image_dir_silhs = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/upi-s1h/trial/images/"
    # train_label_dir_silhs = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/upi-s1h/trial/masks/"
    train_image_dir_segs = "/data/cvfs/as2562/4th_year_proj_datasets/s31_padded_small_glob_rot/images"
    train_label_dir_segs = "/data/cvfs/as2562/4th_year_proj_datasets/s31_padded_small_glob_ro/masks"
    train_image_dir_silhs = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii_padded/images"
    train_label_dir_silhs = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii_padded/masks"

    monitor_dir = "./second_stage_monitor2/monitor_train_images"
    # TODO create validation directory
    num_classes_segs = 32
    num_classes_silhs = 2

    assert os.path.isdir(train_image_dir_segs), 'Invalid segs image directory'
    assert os.path.isdir(train_label_dir_segs), 'Invalid segs label directory'
    assert os.path.isdir(train_image_dir_silhs), 'Invalid silhs image directory'
    assert os.path.isdir(train_label_dir_silhs), 'Invalid silhs label directory'

    segs_image_data_gen_args = dict(
        rescale=(1 / 255.0),
        fill_mode='nearest')

    segs_mask_data_gen_args = dict(
        fill_mode='nearest')

    silhs_image_data_gen_args = dict(
        rescale=(1/255.0),
        fill_mode='nearest')

    silhs_mask_data_gen_args = dict(
        rescale=(1 / 255.0),
        fill_mode='nearest')

    segs_image_datagen = ImageDataGenerator(**segs_image_data_gen_args)
    segs_mask_datagen = ImageDataGenerator(**segs_mask_data_gen_args)
    silhs_image_datagen = ImageDataGenerator(**silhs_image_data_gen_args)
    silhs_mask_datagen = ImageDataGenerator(**silhs_mask_data_gen_args)

    # Provide the same seed to flow methods for train generators
    seed = 1
    segs_image_generator = segs_image_datagen.flow_from_directory(
        train_image_dir_segs,
        batch_size=batch_size,
        target_size=(input_wh, input_wh),
        class_mode=None,
        seed=seed)

    segs_mask_generator = segs_mask_datagen.flow_from_directory(
        train_label_dir_segs,
        batch_size=batch_size,
        target_size=(segs_output_wh, segs_output_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    silhs_image_generator = silhs_image_datagen.flow_from_directory(
        train_image_dir_silhs,
        batch_size=batch_size,
        target_size=(input_wh, input_wh),
        class_mode=None,
        seed=seed)

    silhs_mask_generator = silhs_mask_datagen.flow_from_directory(
        train_label_dir_silhs,
        batch_size=batch_size,
        target_size=(silhs_output_wh, silhs_output_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    print('Generators loaded.')

    # # For testing data loading
    # x = segs_image_generator.next()
    # y = segs_mask_generator.next()
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(x[0, :, :, :])
    # plt.subplot(222)
    # plt.imshow(y[0, :, :, 0])
    # y_post = classlab(y[0], num_classes_segs)
    # plt.subplot(223)
    # plt.imshow(y_post[:, :, 0])
    # plt.subplot(224)
    # plt.imshow(y_post[:, :, 1])
    # plt.show()
    #
    # x = silhs_image_generator.next()
    # y = silhs_mask_generator.next()
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(x[0, :, :, :])
    # plt.subplot(222)
    # plt.imshow(y[0, :, :, 0])
    # y_post = classlab(y[0], num_classes_silhs)
    # plt.subplot(223)
    # plt.imshow(y_post[:, :, 0])
    # plt.subplot(224)
    # plt.imshow(y_post[:, :, 1])
    # plt.show()

    print("Resuming from ", resume_from)
    smpl_model = load_model(os.path.join("./full_network_weights", resume_from),
                            custom_objects={'dd': dd,
                                            'tf': tf})

    verts_model, projects_model, silhouettes_model, segs_model = \
        build_full_model_from_saved_model(smpl_model,
                                          segs_output_wh,
                                          "./neutral_smpl_with_cocoplus_reg.pkl",
                                          batch_size,
                                          num_classes_segs)
    print("Models loaded.")

    adam_optimiser = Adam(lr=0.0001)
    silhouettes_model.compile(optimizer=adam_optimiser,
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
    print("Silhouettes model compiled.")

    segs_model.compile(optimizer=adam_optimiser,
                       loss=categorical_focal_loss(gamma=2.0,
                                                   weight_classes=weight_segs_classes),
                       metrics=['accuracy'])
    print("Segs model compiled.")

    for trial in range(4000):
        print("Fitting", trial)

        def segs_train_data_gen():
            while True:
                train_data, train_labels = generate_data(segs_image_generator,
                                                         segs_mask_generator,
                                                         batch_size,
                                                         num_classes_segs)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, segs_output_wh * segs_output_wh,
                                                    num_classes_segs))
                yield (train_data, reshaped_train_labels)

        def silhs_train_data_gen():
            while True:
                train_data, train_labels = generate_data(silhs_image_generator,
                                                         silhs_mask_generator,
                                                         batch_size,
                                                         num_classes_silhs)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, silhs_output_wh * silhs_output_wh,
                                                    num_classes_silhs))
                yield (train_data, reshaped_train_labels)

        history_segs = segs_model.fit_generator(segs_train_data_gen(),
                                                steps_per_epoch=300,
                                                nb_epoch=1,
                                                verbose=1)

        history_silhs = silhouettes_model.fit_generator(silhs_train_data_gen(),
                                                        steps_per_epoch=150,
                                                        nb_epoch=1,
                                                        verbose=1)

        renderer = SMPLRenderer()

        if trial % 10 == 0:
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
            segs1 = np.reshape(segs_model.predict(input_images_array1),
                               [-1, segs_output_wh, segs_output_wh, num_classes_segs])
            silhs1 = np.reshape(silhouettes_model.predict(input_images_array1),
                                [-1, silhs_output_wh, silhs_output_wh, num_classes_silhs])

            smpls2 = smpl_model.predict(input_images_array2)
            verts2 = verts_model.predict(input_images_array2)
            projects2 = projects_model.predict(input_images_array2)
            segs2 = np.reshape(segs_model.predict(input_images_array2),
                               [-1, segs_output_wh, segs_output_wh, num_classes_segs])
            silhs2 = np.reshape(silhouettes_model.predict(input_images_array2),
                                [-1, silhs_output_wh, silhs_output_wh, num_classes_silhs])

            smpls = np.concatenate((smpls1, smpls2), axis=0)
            verts = np.concatenate((verts1, verts2), axis=0)
            projects = np.concatenate((projects1, projects2), axis=0)
            segs = np.concatenate((segs1, segs2), axis=0)
            silhs = np.concatenate((silhs1, silhs2), axis=0)

            seg_maps = np.argmax(segs, axis=-1)
            silh_maps = np.argmax(silhs, axis=-1)

            print(smpls[0])
            for i in range(smpls.shape[0]):
                plt.figure(1)
                plt.clf()
                plt.imshow(seg_maps[i])
                plt.savefig("./second_stage_monitor/seg_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(2)
                plt.clf()
                plt.imshow(silh_maps[i])
                plt.savefig("./second_stage_monitor/silh_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(3)
                plt.clf()
                plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./second_stage_monitor/verts_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(4)
                rend_img = renderer(verts=verts[i], render_seg=False)
                plt.imshow(rend_img)
                plt.savefig("./second_stage_monitor/rend_" + str(trial) + "_" + str(i) + ".png")

                if trial == 0:
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(input_images_array[i, :, :, :])
                    plt.savefig("./second_stage_monitor/image_" + str(i) + ".png")

            if save_model:
                save_fname = "second_stage48x48_combined_" + str(trial) + "_" + resume_from
                smpl_model.save(os.path.join('./test_models', save_fname))
                print('SAVE NAME', save_fname)

    print("Finished")


train("up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5",
      256,
      48,
      save_model=True,
      weight_segs_classes=True)

