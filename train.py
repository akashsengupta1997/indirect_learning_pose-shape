import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import deepdish as dd

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from renderer import SMPLRenderer
from model import build_model, build_full_model_from_saved_model
from focal_loss import categorical_focal_loss


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


def train(input_wh, output_wh, dataset, num_gpus=1, use_IEF=False, vertex_sampling=None,
          scaledown=0.005, weight_classes=False, save_model=False, resume_from=None):
    batch_size = 4 * num_gpus

    if dataset == 'up-s31':
        # train_image_dir = "/data/cvfs/as2562/4th_year_proj_datasets/s31_padded_small_glob_rot/images"
        # train_label_dir = "/data/cvfs/as2562/4th_year_proj_datasets/s31_padded_small_glob_rot/masks"
        train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/images"
        train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/masks"
        monitor_dir = "./full_network_monitor_train/monitor_train_images"
        # TODO create validation directory
        num_classes = 32
        num_train_images = 5932

    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    # assert os.path.isdir(val_image_dir), 'Invalid validation image directory'
    # assert os.path.isdir(val_label_dir), 'Invalid validation label directory'

    train_image_data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=False,
        rescale=1/255.0,
        fill_mode='nearest')

    train_mask_data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=False,
        fill_mode='nearest')

    val_image_data_gen_args = dict(
        rescale=(1/255.0),
        fill_mode='nearest')

    val_mask_data_gen_args = dict(
        fill_mode='nearest')

    # TODO play with augmentation
    train_image_datagen = ImageDataGenerator(**train_image_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_mask_data_gen_args)
    # val_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    # val_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

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

    # val_image_generator = val_image_datagen.flow_from_directory(
    #     val_image_dir,
    #     batch_size=batch_size,
    #     target_size=(img_wh, img_wh),
    #     class_mode=None,
    #     seed=seed)
    #
    # val_mask_generator = val_mask_datagen.flow_from_directory(
    #     val_label_dir,
    #     batch_size=batch_size,
    #     target_size=(img_dec_wh, img_dec_wh),
    #     class_mode=None,
    #     color_mode="grayscale",
    #     seed=seed)

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
    # y_post = classlab(y[0], num_classes)
    # plt.subplot(223)
    # plt.imshow(y_post[:, :, 0])
    # plt.subplot(224)
    # plt.imshow(y_post[:, :, 28])
    # plt.show()

    adam_optimiser = Adam(lr=0.0001)

    if resume_from is not None:
        print("Resuming model from ", resume_from)
        smpl_model = load_model(os.path.join("./test_models", resume_from),
                                custom_objects={'dd': dd,
                                                'tf': tf})

        verts_model, projects_model, segs_model = build_full_model_from_saved_model(smpl_model,
                                                                                    output_wh,
                                                                                    "./neutral_smpl_with_cocoplus_reg.pkl",
                                                                                    batch_size / num_gpus,
                                                                                    num_classes)
        print("Model loaded.")

    else:
        segs_model, smpl_model, verts_model, projects_model = build_model(
            batch_size / num_gpus,
            (input_wh, input_wh, 3),
            "./neutral_smpl_with_cocoplus_reg.pkl",
            output_wh,
            num_classes,
            use_IEF=use_IEF,
            vertex_sampling=vertex_sampling,
            scaledown=scaledown)

    if num_gpus > 1:
        parallel_segs_model = multi_gpu_model(segs_model, gpus=num_gpus)
        parallel_segs_model.compile(optimizer=adam_optimiser,
                                    loss=categorical_focal_loss(gamma=2.0,
                                                                weight_classes=weight_classes),
                                    metrics=['accuracy'])
    elif num_gpus == 1:
        segs_model.compile(optimizer=adam_optimiser,
                           loss=categorical_focal_loss(gamma=2.0,
                                                       weight_classes=weight_classes),
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

        if num_gpus > 1:
            history = parallel_segs_model.fit_generator(train_data_gen(),
                                                        steps_per_epoch=int(num_train_images/(batch_size*5)),
                                                        nb_epoch=1,
                                                        verbose=1)
        elif num_gpus == 1:
            history = segs_model.fit_generator(train_data_gen(),
                                               steps_per_epoch=int(num_train_images/(batch_size*5)),
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
            input_images_array1 = input_images_array[:batch_size/num_gpus, :, :, :]
            input_images_array2 = input_images_array[batch_size/num_gpus:, :, :, :]

            smpls1 = smpl_model.predict(input_images_array1)
            verts1 = verts_model.predict(input_images_array1)
            projects1 = projects_model.predict(input_images_array1)
            segs1 = np.reshape(segs_model.predict(input_images_array1),
                               [-1, output_wh, output_wh, num_classes])

            smpls2 = smpl_model.predict(input_images_array2)
            verts2 = verts_model.predict(input_images_array2)
            projects2 = projects_model.predict(input_images_array2)
            segs2 = np.reshape(segs_model.predict(input_images_array2),
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
                plt.savefig("./full_network_monitor_train/seg_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(2)
                plt.clf()
                plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./full_network_monitor_train/verts_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(3)
                rend_img = renderer(verts=verts[i], render_seg=False)
                plt.imshow(rend_img)
                plt.savefig("./full_network_monitor_train/rend_" + str(trial) + "_" + str(i) + ".png")

                if trial == 0:
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(input_images_array[i, :, :, :])
                    plt.savefig("./full_network_monitor_train/image_" + str(i) + ".png")

            if save_model:
                save_fname = "{dataset}_{output_wh}x{output_wh}_resnet".format(dataset=dataset,
                                                                               output_wh=output_wh)
                if use_IEF:
                    save_fname += "_ief"
                save_fname += "_scaledown{scaledown}".format(
                    scaledown=str(scaledown).replace('.', ''))
                if vertex_sampling is not None:
                    save_fname += "_vs{vertex_sampling}".format(vertex_sampling=vertex_sampling)
                if weight_classes:
                    save_fname += "_arms_weighted_2_bg_weighted_0point3_gamma2_multigpu"
                save_fname += "_{trial}.hdf5".format(trial=trial)
                smpl_model.save(os.path.join('./test_models', save_fname))
                print('SAVE NAME', save_fname)

    print("Finished")


train(256, 48, 'up-s31', use_IEF=True, vertex_sampling=None, scaledown=0.005,
      weight_classes=True, save_model=True, num_gpus=1, resume_from=None)

