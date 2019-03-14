import os

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2
import pickle

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Lambda

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import orthographic_project2
from keras_smpl.compute_mask import compute_mask
from keras_smpl.projects_to_seg import projects_to_seg

from renderer import SMPLRenderer


def create_input_tensor(image_dir, input_wh, num_classes):
    inputs = []
    fnames = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_seg = cv2.imread(os.path.join(image_dir, fname), 0)
            input_seg = cv2.resize(input_seg, (input_wh, input_wh),
                                   interpolation=cv2.INTER_NEAREST)
            input_seg = np.expand_dims(input_seg, axis=-1)
            input_seg = input_seg * (1.0 / (num_classes - 1))
            inputs.append(input_seg)
            fnames.append(os.path.splitext(fname)[0])

    input_tensor = np.array(inputs)
    print('Input shape:', input_tensor.shape)

    return input_tensor, fnames


def build_full_model(smpl_model, output_wh, smpl_path, batch_size):
    inp = smpl_model.input
    smpl = smpl_model.output
    verts = SMPLLayer(smpl_path, batch_size=batch_size)(smpl)
    projects_with_depth = Lambda(orthographic_project2,
                                 arguments={'vertex_sampling': None},
                                 name='project')([verts, smpl])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': None},
                  name='segment')([projects_with_depth, masks])

    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)
    segs_model = Model(inputs=inp, outputs=segs)

    return verts_model, projects_model, segs_model


def predict_autoencoder(input_wh, output_wh, num_classes, model_fname, save=False):
    test_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks/train'
    input_tensor, fnames = create_input_tensor(test_image_dir, input_wh, num_classes)
    num_inputs = input_tensor.shape[0]

    smpl_model = load_model(os.path.join("./test_models", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model(smpl_model,
                                                               output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl",
                                                               num_inputs)

    smpls = smpl_model.predict(input_tensor)
    verts = verts_model.predict(input_tensor)
    projects = projects_model.predict(input_tensor)
    segs = segs_model.predict(input_tensor)
    seg_maps = np.argmax(segs, axis=-1)
    renderer = SMPLRenderer()

    for i in range(num_inputs):
        plt.figure(1)
        plt.clf()
        plt.imshow(seg_maps[i])
        if save:
            plt.savefig(os.path.join(test_image_dir,
                                     "{fname}_seg.png").format(fname=fnames[i]))
        plt.figure(2)
        plt.clf()
        plt.scatter(projects[i, :, 0], projects[i, :, 1], s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        if save:
            plt.savefig(os.path.join(test_image_dir,
                                     "{fname}_projects.png").format(fname=fnames[i]))
        plt.figure(3)
        rend_img = renderer(verts=verts[i], render_seg=False)
        plt.imshow(rend_img)
        if save:
            plt.savefig(os.path.join(test_image_dir,
                                     "{fname}_rend.png").format(fname=fnames[i]))


predict_autoencoder(256, 80, 32, 'up-s31_80x80_resnet_ief_scaledown0005_0.hdf5',
                    save=True)
