import numpy as np
import os
import cv2
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


def build_model(train_batch_size, input_shape, smpl_path, output_wh, num_classes,
                encoder_architecture='resnet50', use_IEF=False, vertex_sampling=None,
                scaledown=0.005):
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
        state1, param1 = Lambda(concat_mean_param,
                                arguments={'img_wh': output_wh})(img_features)

        # Iteration 1
        delta1 = IEF_layer_1(state1)
        delta1 = IEF_layer_2(delta1)
        delta1 = IEF_layer_3(delta1)
        delta1 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta1)
        param2 = Add()([param1, delta1])
        state2 = Concatenate()([img_features, param2])

        # Iteration 2
        delta2 = IEF_layer_1(state2)
        delta2 = IEF_layer_2(delta2)
        delta2 = IEF_layer_3(delta2)
        delta2 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta2)
        param3 = Add()([param2, delta2])
        state3 = Concatenate()([img_features, param3])

        # Iteration 3
        delta3 = IEF_layer_1(state3)
        delta3 = IEF_layer_2(delta3)
        delta3 = IEF_layer_3(delta3)
        delta3 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta3)
        final_param = Add()([param3, delta3])

    else:
        smpl = Dense(2048, activation='relu')(img_features)
        smpl = Dense(1024, activation='relu')(smpl)
        smpl = Dense(num_total_params, activation='linear')(smpl)
        smpl = Lambda(lambda x: x * scaledown, name="scale_down")(smpl)
        final_param = Lambda(load_mean_set_cam_params,
                             arguments={'img_wh': output_wh})(smpl)

    verts = SMPLLayer(smpl_path, batch_size=train_batch_size)(final_param)
    projects_with_depth = Lambda(orthographic_project2,
                                 arguments={'vertex_sampling': vertex_sampling},
                                 name='project')([verts, final_param])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': vertex_sampling},
                  name='segment')([projects_with_depth, masks])
    segs = Reshape((output_wh * output_wh, num_classes))(segs)
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


def train(input_wh, output_wh, dataset, use_IEF=False, vertex_sampling=None,
          scaledown=0.005, weight_classes=False, save_model=False):
    batch_size = 3

    if dataset == 'up-s31':
        train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images"
        train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
        monitor_dir = "./full_network_monitor_train/monitor_train_images"
        # TODO create validation directory
        num_classes = 32
        # num_train_images = 8515
        # num_train_images = 31
        num_train_images = 3

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
    # plt.imshow(y_post[:, :, 13])
    # plt.show()

    segs_model, smpl_model, verts_model, projects_model = build_model(
        batch_size,
        (input_wh, input_wh, 3),
        "./neutral_smpl_with_cocoplus_reg.pkl",
        output_wh,
        num_classes,
        use_IEF=use_IEF,
        vertex_sampling=vertex_sampling,
        scaledown=scaledown)

    adam_optimiser = Adam(lr=0.0001)
    segs_model.compile(optimizer=adam_optimiser,
                       loss=categorical_focal_loss(gamma=5.0, weight_classes=weight_classes),
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
                                           steps_per_epoch=int(num_train_images/batch_size),
                                           nb_epoch=1,
                                           verbose=1)

        renderer = SMPLRenderer()

        if trial % 50 == 0:
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
            input_images_array = input_images_array[:batch_size, :, :, :]

            smpls = smpl_model.predict(input_images_array)
            verts = verts_model.predict(input_images_array)
            projects = projects_model.predict(input_images_array)
            segs = np.reshape(segs_model.predict(input_images_array),
                              [-1, output_wh, output_wh, num_classes])
            seg_maps = np.argmax(segs, axis=-1)

            print(smpls[0])
            i = 0
            while i < batch_size:
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
                i += 1

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
                save_fname += "_weighted"
            save_fname += "_{trial}.hdf5".format(trial=trial)
            smpl_model.save(os.path.join('./test_models', save_fname))
            print('SAVE NAME', save_fname)

    print("Finished")


train(256, 96, 'up-s31', use_IEF=True, vertex_sampling=None, scaledown=0.005,
      weight_classes=True, save_model=True)

