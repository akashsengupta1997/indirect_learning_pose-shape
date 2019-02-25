import numpy as np
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D, \
    BatchNormalization, Activation, Add, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras.optimizers import Adam

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project2
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.concat_mean_param import concat_mean_param
from keras_smpl.set_cam_params import load_mean_set_cam_params
from keras_smpl.compute_mask import compute_mask

from encoders.encoder_enet_simple import build_enet
from renderer import SMPLRenderer

from focal_loss import categorical_focal_loss


def build_model(train_batch_size, input_shape, smpl_path, output_img_wh, num_classes,
                encoder_architecture='resnet50', use_IEF = False):
    """
    Build indirect learning model, using backbone designated in encoder_architecture.
    :param train_batch_size: batch size to use for training
    :param input_shape: H x W x 3
    :param smpl_path: path to SMPL model data
    :param output_img_wh: width and height of output image
    :param num_classes: number of part segmentation classes
    :param encoder_architecture: backbone architecture
    :return: 4 models - each takes images as inputs and outputs segmentations, smpl params,
    vertex projections and renderings respectively.
    """
    num_camera_params = 4
    num_smpl_params = 72 + 10
    num_total_params = num_smpl_params + num_camera_params

    # --- BACKBONE ---
    if encoder_architecture == 'enet':
        inp = Input(shape=input_shape)
        img_features = build_enet(inp)  # (N, 32, 32, 128) output size from enet

    elif encoder_architecture == 'resnet50':
        resnet = resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape)
        inp = resnet.input
        img_features = resnet.output
        img_features = Reshape((2048,))(img_features)

    if use_IEF:
        # --- IEF MODULE ---
        # Instantiate ief layers
        IEF_layer_1 = Dense(1024, activation='relu', name='IEF_layer_1')
        IEF_layer_2 = Dense(1024, activation='relu', name='IEF_layer_2')
        IEF_layer_3 = Dense(num_total_params, activation='linear', name='IEF_layer_3')

        # Load mean params and set initial state to concatenation of image features and mean params
        state1, param1 = Lambda(concat_mean_param)(img_features)

        # Iteration 1
        delta1 = IEF_layer_1(state1)
        delta1 = IEF_layer_2(delta1)
        delta1 = IEF_layer_3(delta1)
        param2 = Add()([param1, delta1])
        state2 = Concatenate()([img_features, param2])

        # Iteration 2
        delta2 = IEF_layer_1(state2)
        delta2 = IEF_layer_2(delta2)
        delta2 = IEF_layer_3(delta2)
        param3 = Add()([param2, delta2])
        state3 = Concatenate()([img_features, param3])

        # Iteration 3
        delta3 = IEF_layer_1(state3)
        delta3 = IEF_layer_2(delta3)
        delta3 = IEF_layer_3(delta3)
        final_param = Add()([param3, delta3])

    else:
        smpl = Dense(2048, activation='relu')(img_features)
        smpl = Dense(1024, activation='relu')(smpl)
        smpl = Dense(num_total_params, activation='linear')(smpl)
        smpl = Lambda(lambda x: x * 0.1, name="scale_down")(smpl)
        final_param = Lambda(load_mean_set_cam_params)(smpl)

    verts = SMPLLayer(smpl_path, batch_size=train_batch_size)(final_param)
    projects_with_depth = Lambda(orthographic_project2, name='project')([verts, final_param])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg, name='segment')([projects_with_depth, masks])
    segs = Reshape((output_img_wh * output_img_wh, num_classes))(segs)
    segs = Activation('softmax')(segs)

    segs_model = Model(inputs=inp, outputs=segs)
    smpl_model = Model(inputs=inp, outputs=final_param)
    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)

    print(segs_model.summary())

    return segs_model, smpl_model, verts_model, projects_model


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


def train(img_wh, output_img_wh, dataset):
    batch_size = 1  # TODO change back to 10

    if dataset == 'up-s31':
        train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images"
        train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
        # TODO create validation directory
        num_classes = 32
        num_train_images = 8515

    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    # assert os.path.isdir(val_image_dir), 'Invalid validation image directory'
    # assert os.path.isdir(val_label_dir), 'Invalid validation label directory'

    train_image_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1/255.0,
        fill_mode='nearest')

    train_mask_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_image_data_gen_args = dict(
        rescale=(1/255.0),
        fill_mode='nearest')

    val_mask_data_gen_args = dict(
        fill_mode='nearest')

    # TODO add back augmentation
    train_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)
    # val_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    # val_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

    # Provide the same seed to flow methods for train generators
    seed = 1
    train_image_generator = train_image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        target_size=(output_img_wh, output_img_wh),
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

    # For testing data loading
    x = train_image_generator.next()
    y = train_mask_generator.next()
    print('x shape out of training generator', x.shape)  # should = (batch_size, img_hw, img_hw, 3)
    print('y shape out of training generator', y.shape)  # should = (batch_size, dec_hw, dec_hw, 1)
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(x[0, :, :, :])
    plt.subplot(222)
    plt.imshow(y[0, :, :, 0])
    y_post = classlab(y[0], num_classes)
    plt.subplot(223)
    plt.imshow(y_post[:, :, 0])
    plt.subplot(224)
    plt.imshow(y_post[:, :, 13])
    plt.show()

    segs_model, smpl_model, verts_model, projects_model = build_model(1,
                                       (img_wh, img_wh, 3),
                                       "./neutral_smpl_with_cocoplus_reg.pkl",
                                       output_img_wh,
                                       num_classes)
    # adam_optimiser = Adam(lr=0.0005)
    # segs_model.compile(loss='categorical_crossentropy',
    #                              optimizer=adam_optimiser,
    #                              metrics=['accuracy'])
    segs_model.compile(optimizer="adam",
                                 loss=categorical_focal_loss(gamma=5.0),
                                 metrics=['accuracy'])


    print("Model compiled.")

    for trials in range(4000):
        nb_epoch = 1
        print("Fitting", trials)

        def train_data_gen():
            while True:
                train_data, train_labels = generate_data(train_image_generator,
                                                         train_mask_generator,
                                                         batch_size,
                                                         num_classes,
                                                         dataset)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, output_img_wh * output_img_wh,
                                                    num_classes))
                yield (train_data, reshaped_train_labels)

        # def val_data_gen():
        #     while True:
        #         val_data, val_labels = generate_data(val_image_generator,
        #                                                  val_mask_generator,
        #                                                  batch_size, num_classes)
        #         reshaped_val_labels = np.reshape(val_labels,
        #                                            (batch_size, img_dec_wh * img_dec_wh,
        #                                             num_classes))
        #         yield (val_data, reshaped_val_labels)

        history = segs_model.fit_generator(train_data_gen(),
                                            steps_per_epoch=1,
                                            nb_epoch=nb_epoch,
                                            verbose=1)

        if trials % 20 == 0:
            # TODO remove this testing code
            test_data, test_gt = generate_data(train_image_generator,
                                               train_mask_generator,
                                               1,
                                               num_classes)
            print(smpl_model.predict(test_data))
            test_verts = verts_model.predict(test_data)
            test_projects = projects_model.predict(test_data)
            test_seg = np.reshape(segs_model.predict(test_data),
                                  (1, output_img_wh, output_img_wh, num_classes))
            test_seg_map = np.argmax(test_seg[0], axis=-1)
            test_gt_seg_map = np.argmax(np.reshape(test_gt[0],
                                                   (output_img_wh, output_img_wh,
                                                    num_classes)), axis=-1)
            renderer = SMPLRenderer()
            rend_img_keras_model = renderer(verts=test_verts[0], render_seg=False)
            plt.figure(1)
            plt.clf()
            plt.imshow(rend_img_keras_model)
            plt.savefig("./test_outputs/rend_" + str(trials) + ".png")
            plt.figure(2)
            plt.clf()
            plt.scatter(test_projects[0, :, 0], test_projects[0, :, 1], s=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("./test_outputs/verts_" + str(trials) + ".png")
            plt.figure(3)
            plt.clf()
            plt.imshow(test_seg_map)
            plt.savefig("./test_outputs/seg_" + str(trials) + ".png")

            if trials == 0:
                plt.figure(5)
                plt.clf()
                plt.imshow(test_gt_seg_map)
                plt.savefig("./test_outputs/gt_seg.png")

            # plt.show()

        # if trials % 100 == 0:
        #     segs_model.save('test_models/ups31_'
        #                      + str(nb_epoch * (trials + 1)).zfill(4) + '.hdf5')

    print("Finished")

train(256, 96, 'up-s31')
