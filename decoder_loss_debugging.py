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
from keras_smpl.projection import persepective_project, orthographic_project
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.add_mean_params import add_mean_params
from keras_smpl.load_mean_param import load_mean_param, concat_mean_param
from encoders.encoder_enet_simple import build_enet
from renderer import SMPLRenderer


def build_model(train_batch_size, input_shape, smpl_path, output_img_wh, num_classes,
                encoder_architecture='resnet50'):
    # num_camera_params = 5
    num_smpl_params = 72 + 10

    # --- BACKBONE ---
    if encoder_architecture == 'enet':
        inp = Input(shape=input_shape)
        img_features = build_enet(inp)  # (N, 32, 32, 128) output size from enet

    elif encoder_architecture == 'resnet50':
        resnet = resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape)
        inp = resnet.input
        img_features = resnet.output

        print('resnet shape')
        print(img_features.get_shape())
        # img_features = Flatten()(img_features)
        img_features = Reshape((2048,))(img_features)
        print('post reshape shape')
        print(img_features.get_shape())

    # --- IEF MODULE ---
    # Instantiate ief layers
    IEF_layer_1 = Dense(1024, activation='relu', name='IEF_layer_1')
    IEF_layer_2 = Dense(1024, activation='relu', name='IEF_layer_2')
    IEF_layer_3 = Dense(num_smpl_params, activation='linear', name='IEF_layer_3')

    # Load mean params and set initial state to concatenation of image features and mean params
    state1, param1 = Lambda(concat_mean_param)(img_features)
    print('sanity check (same as above')
    print(img_features.get_shape())
    print('mean params shape')
    print(param1.get_shape())
    print('state1 shape')
    print(state1.get_shape())

    # Iteration 1
    delta1 = IEF_layer_1(state1)
    delta1 = IEF_layer_2(delta1)
    delta1 = IEF_layer_3(delta1)
    param2 = Add()([param1, delta1])
    state2 = Concatenate()([img_features, param2])
    print('param2 shape')
    print(param2.get_shape())
    print('state2 shape')
    print(state2.get_shape())

    # Iteration 2
    delta2 = IEF_layer_1(state2)
    delta2 = IEF_layer_2(delta2)
    delta2 = IEF_layer_3(delta2)
    param3 = Add()([param2, delta2])
    state3 = Concatenate()([img_features, param3])
    print('param3 shape')
    print(param3.get_shape())
    print('state3 shape')
    print(state3.get_shape())

    # Iteration 3
    delta3 = IEF_layer_1(state3)
    delta3 = IEF_layer_2(delta3)
    delta3 = IEF_layer_3(delta3)
    final_param = Add()([param3, delta3])
    print('final param shape')
    print(final_param.get_shape())

    # encoder = Dense(2048, activation='relu')(img_features)
    # encoder = BatchNormalization()(encoder)
    # encoder = Dense(1024, activation='relu')(encoder)
    # encoder = BatchNormalization()(encoder)
    # smpl = Dense(num_smpl_params, activation='tanh')(encoder)
    # # smpl = Lambda(add_mean_params)(smpl)

    verts = SMPLLayer(smpl_path, batch_size=train_batch_size)(final_param)
    # projects = Lambda(persepective_project, name='projection')([verts, smpl])
    projects = Lambda(orthographic_project, name='projection')(verts)
    segs = Lambda(projects_to_seg, name='segmentation')(projects)
    segs = Reshape((output_img_wh * output_img_wh, num_classes))(segs)
    segs = Activation('softmax')(segs)

    segs_model = Model(inputs=inp, outputs=segs)
    smpl_model = Model(inputs=inp, outputs=final_param)
    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects)

    print(segs_model.summary())

    return segs_model, smpl_model, verts_model, projects_model