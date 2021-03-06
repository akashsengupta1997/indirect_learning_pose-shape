import numpy as np
import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt
import deepdish as dd

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from generators.image_generator_with_fname import ImagesWithFnames

from renderer import SMPLRenderer

from focal_loss import categorical_focal_loss

from model import build_model, build_full_model_from_saved_model


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


def generate_data(input_mask_generator, output_mask_generator, n, num_classes):
    input_labels = []
    output_seg_labels = []
    output_joints_labels = []
    joints_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-3d/"
    i = 0
    while i < n:
        x, fnames = input_mask_generator.next()
        y = output_mask_generator.next()
        j = 0
        while j < y.shape[0]:
            print(i, j)
            # input_labels.append(classlab(x[j, :, :, :].astype(np.uint8), num_classes))
            # fname = fnames[j]
            # id = fname[6:11]
            # joints_path = os.path.join(joints_dir, id + "_joints.npy")
            # joints = np.load(joints_path)

            input_labels.append(x[j, :, :, :])
            output_seg_labels.append(classlab(y[j, :, :, :].astype(np.uint8), num_classes))
            j = j + 1
            i = i + 1
            if i >= n:
                break
    print("Shape in generate_data", np.array(input_labels).shape, np.array(output_seg_labels).shape)
    return np.array(input_labels), np.array(output_seg_labels)


def train(input_wh, output_wh, dataset, num_gpus=1, use_IEF=False, vertex_sampling=None,
          scaledown=0.005, weight_classes=False, save_model=False, resume_from=None):
    batch_size = 4 * num_gpus

    if dataset == 'up-s31':
        # train_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
        train_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/masks"
        # val_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
        monitor_dir = "./monitor_train4/monitor_train_images"
        num_classes = 32
        num_train_images = 5932
        # num_train_images = 3

    assert os.path.isdir(train_dir), 'Invalid train directory'
    # assert os.path.isdir(val_dir), 'Invalid validation directory'

    train_mask_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_input_mask_data_gen_args = dict(
        fill_mode='nearest',
        rescale=(1.0/(num_classes-1)))

    val_mask_data_gen_args = dict(
        fill_mode='nearest')

    # TODO add back augmentation
    train_input_mask_datagen = ImageDataGenerator(**val_input_mask_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)
    # val_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    # val_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

    seed = 1
    # input_mask_generator = train_input_mask_datagen.flow_from_directory(
    #     train_dir,
    #     batch_size=batch_size,
    #     target_size=(input_wh, input_wh),
    #     class_mode=None,
    #     color_mode="grayscale",
    #     seed=seed)

    output_mask_generator = train_mask_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        target_size=(output_wh, output_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    input_mask_generator = ImagesWithFnames(train_dir,
                                            train_input_mask_datagen,
                                            batch_size=batch_size,
                                            target_size=(input_wh, input_wh),
                                            class_mode=None,
                                            color_mode='grayscale',
                                            seed=seed)

    print('Generators loaded.')

    # # For testing data loading
    # x, fnames = input_mask_generator.next()
    # y = output_mask_generator.next()
    #
    # print(fnames)
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(x[0, :, :, 0])
    # plt.subplot(222)
    # plt.imshow(y[0, :, :, 0])
    # y_post = classlab(y[0], num_classes)
    # plt.subplot(223)
    # plt.imshow(y_post[:, :, 0])
    # plt.subplot(224)
    # plt.imshow(y_post[:, :, 13])

    adam_optimiser = Adam(lr=0.0001)

    if resume_from is not None:
        print("Resuming model from ", resume_from)
        smpl_model = load_model(os.path.join("./autoencoder_models", resume_from),
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
            (input_wh, input_wh, 1),
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
                train_input_labels, train_output_labels = generate_data(input_mask_generator,
                                                                        output_mask_generator,
                                                                        batch_size,
                                                                        num_classes)
                reshaped_output_labels = np.reshape(train_output_labels,
                                                    (batch_size, output_wh * output_wh,
                                                     num_classes))
                yield (train_input_labels, reshaped_output_labels)

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
                    input_labels = cv2.imread(os.path.join(monitor_dir, fname), 0)
                    input_labels = cv2.resize(input_labels, (input_wh, input_wh),
                                              interpolation=cv2.INTER_NEAREST)
                    input_labels = np.expand_dims(input_labels, axis=-1)
                    input_labels = input_labels * (1.0/(num_classes-1))
                    inputs.append(input_labels)

            input_mask_array = np.array(inputs)
            input_mask_array1 = input_mask_array[:batch_size/num_gpus, :, :, :]
            input_mask_array2 = input_mask_array[batch_size/num_gpus:, :, :, :]

            smpls1 = smpl_model.predict(input_mask_array1)
            verts1 = verts_model.predict(input_mask_array1)
            projects1 = projects_model.predict(input_mask_array1)
            segs1 = np.reshape(segs_model.predict(input_mask_array1),
                               [-1, output_wh, output_wh, num_classes])

            smpls2 = smpl_model.predict(input_mask_array2)
            verts2 = verts_model.predict(input_mask_array2)
            projects2 = projects_model.predict(input_mask_array2)
            segs2 = np.reshape(segs_model.predict(input_mask_array2),
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
                plt.savefig("./monitor_train4/seg_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(2)
                plt.clf()
                plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./monitor_train4/verts_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(3)
                rend_img = renderer(verts=verts[i], render_seg=False)
                plt.imshow(rend_img)
                plt.savefig("./monitor_train4/rend_" + str(trial) + "_" + str(i) + ".png")

                if trial == 0:
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(input_mask_array[i, :, :, 0])
                    plt.savefig("./monitor_train4/gt_seg_" + str(i) + ".png")

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
                smpl_model.save(os.path.join('./autoencoder_models', save_fname))
                print('SAVE NAME', save_fname)


train(256, 64, 'up-s31', use_IEF=True, vertex_sampling=None, scaledown=0.005,
      weight_classes=True, save_model=False, resume_from=None)
