import numpy as np

from keras.models import Model
from keras.layers import Input, Flatten, Dense
from matplotlib import pyplot as plt

from keras_smpl.batch_smpl import SMPLLayer
from encoders.encoder_enet_simple import build_enet
from renderer import SMPLRenderer


def build_model(train_batch_size, input_shape, smpl_path):
    num_smpl_params = 3 + 72 + 10
    inp = Input(shape=input_shape)
    enet = build_enet(inp)
    enet = Flatten()(enet)
    smpl = Dense(num_smpl_params)(enet)
    verts = SMPLLayer(smpl_path, batch_size=train_batch_size)(smpl)
    model = Model(inputs=inp, outputs=verts)

    # TODO add verts to seg layer

    return model


def convert_to_seg_predict(model):
    """
    Converts training indirect learning model to test indirect learning model that outputs
    bodypart segmentations
    :param model:
    :return:
    """
    # TODO pop off lambda layer, SMPL layer, add SMPL layer with batch_size = 1
    pass


def convert_to_SMPL_predict(model):
    """
    Converts training indirect learning model to test model that outputs SMPL params.
    :param model:
    :return:
    """
    # TODO pop off lambda layer and SMPL layer
