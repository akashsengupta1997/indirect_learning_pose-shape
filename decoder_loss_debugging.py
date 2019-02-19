import numpy as np
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, Activation, Embedding
from keras.preprocessing import image
from keras.applications import resnet50
from keras.optimizers import Adam

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project, \
    orthographic_project2
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.set_cam_params import set_cam_params, load_mean_set_cam_params
from keras_smpl.load_mean_param import load_mean_param, concat_mean_param
from encoders.encoder_enet_simple import build_enet
from renderer import SMPLRenderer

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


def load_masks_from_indices(indices, output_shape):
    masks_folder = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/masks/train"
    labels = []
    for index in list(indices):
        mask_name = str(index).zfill(5) + "_ann.png"
        mask_path = os.path.join(masks_folder, mask_name)
        mask = image.load_img(mask_path, grayscale=True, target_size=output_shape)
        mask = image.img_to_array(mask)
        OHE_label = classlab(mask, 32)
        labels.append(OHE_label)

    # labels = np.array(labels)
    # print(labels.shape)
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(labels[0, :, :, 0])
    # plt.subplot(222)
    # plt.imshow(labels[0, :, :, 10])
    # plt.subplot(223)
    # plt.imshow(labels[0, :, :, 20])
    # plt.subplot(224)
    # plt.imshow(labels[0, :, :, 30])
    # plt.show()

    return labels


def build_debug_model(batch_size, smpl_path, output_img_wh, num_classes):
    num_camera_params = 4
    num_smpl_params = 72 + 10
    num_total_params = num_smpl_params + num_camera_params

    index_inputs = Input(shape=(1,))
    smpls = Embedding(10, num_total_params, input_length=1)(index_inputs)
    smpls = Lambda(lambda smpls: K.squeeze(smpls, axis=1))(smpls)
    smpls = Lambda(load_mean_set_cam_params)(smpls)

    verts = SMPLLayer(smpl_path, batch_size=batch_size)(smpls)
    # projects = Lambda(persepective_project, name='projection')([verts, smpl])
    projects = Lambda(orthographic_project2, name='projection')([verts, smpls])
    segs = Lambda(projects_to_seg, name='segmentation')(projects)
    segs = Reshape((output_img_wh * output_img_wh, num_classes))(segs)
    segs = Activation('softmax')(segs)

    segs_model = Model(inputs=index_inputs, outputs=segs)
    smpl_model = Model(inputs=index_inputs, outputs=smpls)
    verts_model = Model(inputs=index_inputs, outputs=verts)
    projects_model = Model(inputs=index_inputs, outputs=projects)

    print(segs_model.summary())

    return segs_model, smpl_model, verts_model, projects_model


def train(output_wh, num_classes, num_indices):
    # train_indices = np.arange(num_indices)
    train_indices = np.array([2, 10, 11])
    labels = load_masks_from_indices(train_indices, (output_wh, output_wh))
    train_labels = np.reshape(labels, (-1, output_wh*output_wh, num_classes))
    segs_model, smpl_model, verts_model, projects_model = build_debug_model(num_indices,
                                                                            "./neutral_smpl_with_cocoplus_reg.pkl",
                                                                            output_wh,
                                                                            num_classes)

    segs_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    # segs_model.compile(optimizer="adam", loss=categorical_focal_loss(gamma=2.0), metrics=['accuracy'])

    for trial in range(1601):
        print "Epoch", trial
        segs_model.fit(train_indices, train_labels, batch_size=num_indices, verbose=1)

        renderer = SMPLRenderer()

        if trial % 100 == 0:
            test_segs = segs_model.predict(train_indices, batch_size=num_indices)
            test_projects = projects_model.predict(train_indices, batch_size=num_indices)
            test_smpls = smpl_model.predict(train_indices, batch_size=num_indices)
            test_verts = verts_model.predict(train_indices, batch_size=num_indices)
            print(test_smpls[0])
            for idx in range(num_indices):
                test_seg = np.argmax(np.reshape(test_segs[idx], (output_wh, output_wh, num_classes)), axis=-1)
                plt.figure(1)
                plt.clf()
                plt.imshow(test_seg)
                plt.savefig("./test_outputs/seg_" + str(trial) + "_" + str(idx) + ".png")
                plt.figure(2)
                plt.clf()
                plt.scatter(test_projects[idx, :, 0], test_projects[idx, :, 1], s=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("./test_outputs/verts_" + str(trial) + "_" + str(idx) + ".png")
                plt.figure(3)
                rend_img = renderer(verts=test_verts[idx], render_seg=False)
                plt.imshow(rend_img)
                plt.savefig("./test_outputs/rend_" + str(trial) + "_" + str(idx) + ".png")

                if trial == 0:
                    gt_seg = np.argmax(labels[idx], axis=-1)
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(gt_seg)
                    plt.savefig("./test_outputs/gt_seg_" + str(idx) + ".png")



train(128, 32, 3)