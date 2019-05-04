import os
import time

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


def load_input_seg(image_dir, fname, input_wh, num_classes):
    input_img = cv2.imread(os.path.join(image_dir, fname))
    input_img = cv2.resize(input_img, (input_wh, input_wh))
    input_img = input_img[..., ::-1]
    input_img = input_img * (1.0 / 255)
    input_img = np.expand_dims(input_img, axis=0)  # need 4D input (add batch dimension)
    return input_img


def visualise_and_save(fname, verts, projects, seg_maps, renderer, save, save_dir=None):
    plt.figure(1)
    plt.clf()
    plt.imshow(seg_maps[0])
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_seg.png").format(fname=fname))
    plt.figure(2)
    plt.clf()
    plt.scatter(projects[0, :, 0], projects[0, :, 1], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_projects.png").format(fname=fname))
    plt.figure(3)
    rend_img = renderer(verts=verts[0], render_seg=False)
    plt.imshow(rend_img)
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_rend.png").format(fname=fname))
    else:
        plt.show()


def build_full_model(smpl_model, output_wh, smpl_path, batch_size=1):
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


def predict(input_wh, output_wh, num_classes, model_fname, save=False,
                        save_dir=None):
    test_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images/train'
    renderer = SMPLRenderer()
    smpl_model = load_model(os.path.join("./full_network_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model(smpl_model,
                                                               output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl")

    verts_pred_times = []
    segs_pred_times = []

    for fname in sorted(os.listdir(test_image_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_seg = load_input_seg(test_image_dir, fname, input_wh, num_classes)

            start = time.time()
            verts = verts_model.predict(input_seg)
            verts_pred_times.append(time.time()-start)

            projects = projects_model.predict(input_seg)

            start = time.time()
            segs = segs_model.predict(input_seg)
            segs_pred_times.append(time.time() - start)
            seg_maps = np.argmax(segs, axis=-1)

            visualise_and_save(fname, verts, projects, seg_maps, renderer, save=save, save_dir=save_dir)
            print("Average vertices predict time:", np.mean(verts_pred_times))
            print("Average segmentation predict time:", np.mean(segs_pred_times))


predict(256, 48, 32, 'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5',
                    save=True,
                    save_dir='./full_network_test/')
