import os
import time

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Lambda

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import orthographic_project2
from keras_smpl.compute_mask import compute_mask
from keras_smpl.projects_to_seg import projects_to_seg

from renderer import SMPLRenderer
from preprocessing import pad_image


def load_input_img(image_dir, fname, input_wh, pad=False):
    input_img = cv2.imread(os.path.join(image_dir, fname))
    if pad:
        input_img = pad_image(input_img)
    input_img = cv2.resize(input_img, (input_wh, input_wh))
    input_img = input_img[..., ::-1]
    input_img = input_img * (1.0 / 255)
    input_img = np.expand_dims(input_img, axis=0)  # need 4D input (add batch dimension)
    return input_img


def visualise_and_save(fname, padded_img, verts, projects, seg_maps, renderer, input_wh,
                       output_wh, save=False, overlay_projects=False, save_dir=None):
    fname, _ = os.path.splitext(fname)
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
    plt.clf()
    rend_img = renderer(verts=verts[0], render_seg=False)
    plt.imshow(rend_img)
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_rend.png").format(fname=fname))
    plt.figure(4)
    plt.clf()
    plt.imshow(padded_img[0])
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_input.png").format(fname=fname))
    if overlay_projects:
        plt.figure(5)
        plt.clf()
        scatter_scale = float(input_wh)/output_wh
        print(scatter_scale)
        plt.scatter(projects[0, :, 0]*scatter_scale, projects[0, :, 1]*scatter_scale, s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.imshow(cv2.flip(cv2.resize(padded_img[0], (input_wh, input_wh)), 0),
                   alpha=0.9)
        plt.gca().invert_yaxis()
        if save:
            plt.savefig(os.path.join(save_dir,
                                     "{fname}_verts_overlay.png").format(fname=fname))
    if not save:
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


def predict(test_image_dir, input_wh, output_wh, model_fname, save=False, save_dir=None,
            pad=False, overlay_projects=True):
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
        if fname.endswith(".jpg") or fname.endswith(".png"):  # set to jpg or png appropriately
            print(fname)
            input_img = load_input_img(test_image_dir, fname, input_wh, pad=pad)

            start = time.time()
            verts = verts_model.predict(input_img)
            verts_pred_times.append(time.time()-start)

            projects = projects_model.predict(input_img)

            start = time.time()
            segs = segs_model.predict(input_img)
            segs_pred_times.append(time.time() - start)
            seg_maps = np.argmax(segs, axis=-1)

            visualise_and_save(fname, input_img, verts, projects, seg_maps, renderer, input_wh,
                               output_wh, save=save, overlay_projects=overlay_projects,
                               save_dir=save_dir)
            print("Average vertices predict time:", np.mean(verts_pred_times))
            print("Average segmentation predict time:", np.mean(segs_pred_times))


predict('/Users/Akash_Sengupta/Documents/4th_year_project_datasets/my_singleperson_data/imgs',
        256,
        48,
        'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5',
        save=True,
        save_dir='./full_network_test/',
        pad=True,
        overlay_projects=True)
