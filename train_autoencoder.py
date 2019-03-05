import numpy as np
import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D, \
    BatchNormalization, Activation, Add, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project2
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.concat_mean_param import concat_mean_param
from keras_smpl.set_cam_params import load_mean_set_cam_params
from keras_smpl.compute_mask import compute_mask

from generators.image_generator_with_fname import ImagesWithFnames

from encoders.encoder_enet_simple import build_enet
from renderer import SMPLRenderer

from focal_loss import categorical_focal_loss


def build_autoencoder(train_batch_size, input_shape, smpl_path, output_img_wh, num_classes,
                      encoder_architecture='resnet50', use_IEF=False):
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

    elif encoder_architecture == 'simple':
        inp = Input(shape=input_shape)  # using 64 x 64 x 32 input with simple encoder for now

        block1 = Conv2D(256, 3, padding='same')(inp)
        block1 = BatchNormalization()(block1)
        block1 = Activation('relu')(block1)
        block1 = MaxPooling2D()(block1)

        block2 = Conv2D(256, 3, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('relu')(block2)
        block2 = MaxPooling2D()(block2)

        block3 = Conv2D(512, 3, padding='same')(block2)
        block3 = BatchNormalization()(block3)
        block3 = Activation('relu')(block3)
        block3 = MaxPooling2D()(block3)

        block4 = Conv2D(512, 3, padding='same')(block3)
        block4 = BatchNormalization()(block4)
        block4 = Activation('relu')(block4)
        block4 = MaxPooling2D()(block4)

        block5 = Conv2D(1024, 3, padding='same')(block4)
        block5 = BatchNormalization()(block5)
        block5 = Activation('relu')(block5)
        block5 = MaxPooling2D()(block5)

        block6 = Conv2D(2048, 3, padding='same')(block5)
        block6 = BatchNormalization()(block6)
        block6 = Activation('relu')(block6)
        block6 = MaxPooling2D()(block6)
        img_features = Reshape((2048,))(block6)

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
        delta1 = Lambda(lambda x: x * 0.03)(delta1)
        param2 = Add()([param1, delta1])
        state2 = Concatenate()([img_features, param2])

        # Iteration 2
        delta2 = IEF_layer_1(state2)
        delta2 = IEF_layer_2(delta2)
        delta2 = IEF_layer_3(delta2)
        delta2 = Lambda(lambda x: x * 0.03)(delta2)
        param3 = Add()([param2, delta2])
        state3 = Concatenate()([img_features, param3])

        # Iteration 3
        delta3 = IEF_layer_1(state3)
        delta3 = IEF_layer_2(delta3)
        delta3 = IEF_layer_3(delta3)
        delta3 = Lambda(lambda x: x * 0.03)(delta3)
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

    return np.array(input_labels), np.array(output_seg_labels)


def train(input_wh, output_wh, dataset, multi_gpu=False):
    batch_size = 3

    if dataset == 'up-s31':
        train_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks2"
        # train_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/masks"
        # val_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
        monitor_dir = "./monitor_train/monitor_train_images"
        # TODO create validation directory
        num_classes = 32
        # num_train_images = 6813
        num_train_images = 31

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
    if multi_gpu:
        segs_model, smpl_model, verts_model, projects_model = build_autoencoder(
            batch_size,
            # (input_wh, input_wh, num_classes),
            (input_wh, input_wh, 1),
            "./neutral_smpl_with_cocoplus_reg.pkl",
            output_wh,
            num_classes)
        parallel_segs_model = multi_gpu_model(segs_model, gpus=2)
        parallel_segs_model.compile(optimizer=adam_optimiser,
                                    loss=categorical_focal_loss(gamma=5.0),
                                    metrics=['accuracy'])
    else:
        segs_model, smpl_model, verts_model, projects_model = build_autoencoder(
            batch_size,
            (input_wh, input_wh, 1),
            "./neutral_smpl_with_cocoplus_reg.pkl",
            output_wh,
            num_classes)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        segs_model.compile(optimizer=adam_optimiser,
                           loss=categorical_focal_loss(gamma=5.0),
                           metrics=['accuracy'],
                           options=run_options,
                           run_metadata=run_metadata)

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

        if multi_gpu:
            history = parallel_segs_model.fit_generator(
                train_data_gen(),
                steps_per_epoch=int(num_train_images/batch_size),
                nb_epoch=1,
                verbose=1)
        else:
            history = segs_model.fit_generator(train_data_gen(),
                                               steps_per_epoch=int(num_train_images/batch_size),
                                               nb_epoch=1,
                                               verbose=1)

            from tensorflow.python.client import timeline
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        renderer = SMPLRenderer()
        if trial % 50 == 0:
            #
            # inputs = []
            # for fname in sorted(os.listdir(monitor_dir)):
            #     if fname.endswith(".png"):
            #         input_labels = cv2.imread(os.path.join(monitor_dir, fname), 0)
            #         input_labels = cv2.resize(input_labels, (input_wh, input_wh),
            #                                   interpolation=cv2.INTER_NEAREST)
            #         input_labels = np.expand_dims(input_labels, axis=-1)
            #         input_labels = input_labels * (1.0/(num_classes-1))
            #         inputs.append(input_labels)
            #
            # input_mask_array = np.array(inputs)
            # input_mask_array = input_mask_array[:batch_size, :, :, :]
            #
            # smpls = smpl_model.predict(input_mask_array)
            # verts = verts_model.predict(input_mask_array)
            # projects = projects_model.predict(input_mask_array)
            # segs = np.reshape(segs_model.predict(input_mask_array),
            #                   [-1, output_wh, output_wh, num_classes])
            # seg_maps = np.argmax(segs, axis=-1)
            #
            # print(smpls[0])
            # i = 0
            # while i < batch_size:
            #     plt.figure(1)
            #     plt.clf()
            #     plt.imshow(seg_maps[i])
            #     plt.savefig("./monitor_train/seg_" + str(trial) + "_" + str(i) + ".png")
            #     plt.figure(2)
            #     plt.clf()
            #     plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
            #     plt.gca().set_aspect('equal', adjustable='box')
            #     plt.savefig("./monitor_train/verts_" + str(trial) + "_" + str(i) + ".png")
            #     plt.figure(3)
            #     rend_img = renderer(verts=verts[i], render_seg=False)
            #     plt.imshow(rend_img)
            #     plt.savefig("./monitor_train/rend_" + str(trial) + "_" + str(i) + ".png")
            #
            #     if trial == 0:
            #         plt.figure(4)
            #         plt.clf()
            #         plt.imshow(input_mask_array[i, :, :, 0])
            #         plt.savefig("./monitor_train/gt_seg_" + str(i) + ".png")
            #     i += 1

            # TODO remove this testing code
            test_input_labels, test_output_labels = generate_data(input_mask_generator,
                                                                  output_mask_generator,
                                                                  2,
                                                                  num_classes)

            print(smpl_model.predict(test_input_labels)[0])
            test_verts = verts_model.predict(test_input_labels)
            test_projects = projects_model.predict(test_input_labels)
            test_seg = np.reshape(segs_model.predict(test_input_labels),
                                  (-1, output_wh, output_wh, num_classes))
            test_seg_maps = np.argmax(test_seg, axis=-1)
            test_gt_seg_maps = np.argmax(np.reshape(test_output_labels,
                                                   (-1, output_wh, output_wh,
                                                    num_classes)), axis=-1)

            for i in range(batch_size):
                rend_img_keras_model = renderer(verts=test_verts[i], render_seg=False)
                plt.figure(1)
                plt.clf()
                plt.imshow(rend_img_keras_model)
                plt.savefig("./test_outputs/rend_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(2)
                plt.clf()
                plt.scatter(test_projects[i, :, 0], test_projects[i, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./test_outputs/verts_" + str(trial) + "_" + str(i) + ".png")
                plt.figure(3)
                plt.clf()
                plt.imshow(test_seg_maps[i])
                plt.savefig("./test_outputs/seg_" + str(trial) + "_" + str(i) + ".png")

                if trial == 0:
                    plt.figure(5)
                    plt.clf()
                    plt.imshow(test_gt_seg_maps[i])
                    plt.savefig("./test_outputs/gt_seg" + "_" + str(i) + ".png")


        # if trial % 100 == 0:
        #     segs_model.save('test_models/ups31_'
        #                      + str(nb_epoch * (trial + 1)).zfill(4) + '.hdf5')

    print("Finished")


train(256, 96, 'up-s31')
