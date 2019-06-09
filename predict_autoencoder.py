import os
import time

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2

import tensorflow as tf
from keras.models import load_model

from model import build_full_model_for_predict
from renderer import SMPLRenderer
from preprocessing import pad_image


def load_input_seg(image_dir, fname, input_wh, num_classes):
    input_seg = cv2.imread(os.path.join(image_dir, fname), 0)
    input_seg = cv2.resize(input_seg, (input_wh, input_wh),
                           interpolation=cv2.INTER_NEAREST)
    input_seg = np.expand_dims(input_seg, axis=-1)
    input_seg = input_seg * (1.0 / (num_classes - 1))
    input_seg = np.expand_dims(input_seg, axis=0)  # need 4D input (add batch dimension)
    return input_seg


def visualise_and_save(fname, padded_img, verts, projects, seg_maps, renderer, input_wh,
                       output_wh, save=False, save_dir=None, overlay_projects=False):
    fname, _ = os.path.splitext(fname)
    # fname = fname[:5]
    plt.figure(1)
    plt.clf()
    plt.imshow(seg_maps[0])
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_seg.png").format(fname=fname),
                    bbox_inches='tight', pad_inches=0)
    plt.figure(2)
    plt.clf()
    plt.scatter(projects[0, :, 0], projects[0, :, 1], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_projects.png").format(fname=fname),
                    bbox_inches='tight', pad_inches=0)
    plt.figure(3)
    rend_img = renderer(verts=verts[0], render_seg=False)
    plt.imshow(rend_img)
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_rend.png").format(fname=fname),
                    bbox_inches='tight', pad_inches=0)
    plt.figure(4)
    plt.clf()
    plt.imshow(padded_img)
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "{fname}_orig_img.png").format(fname=fname),
                    bbox_inches='tight', pad_inches=0)
    if overlay_projects:
        plt.figure(5)
        plt.clf()
        scatter_scale = float(input_wh) / output_wh
        plt.scatter(projects[0, :, 0] * scatter_scale, projects[0, :, 1] * scatter_scale,
                    s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.imshow(cv2.flip(cv2.resize(padded_img, (input_wh, input_wh)), 0),
                   alpha=0.9)
        plt.gca().invert_yaxis()
        if save:
            plt.savefig(os.path.join(save_dir,
                                     "{fname}_verts_overlay.png").format(fname=fname),
                        bbox_inches='tight', pad_inches=0)
    if not save:
        plt.show()


def predict_autoencoder(input_wh, output_wh, num_classes, model_fname, save=False,
                        save_dir=None, overlay_projects=True, pad_orig_img=True):
    test_image_dir = './results/my_singleperson_imgs/legs_ambiguous/autoencoder_fpn48x48'
    orig_image_dir = './results/my_singleperson_imgs/legs_ambiguous/'
    renderer = SMPLRenderer()
    smpl_model = load_model(os.path.join("./autoencoder_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model_for_predict(smpl_model,
                                                                           output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl")

    verts_pred_times = []
    segs_pred_times = []

    for fname in sorted(os.listdir(test_image_dir)):
        if fname.endswith(".png"):
            print(fname)
            input_seg = load_input_seg(test_image_dir, fname, input_wh, num_classes)
            orig_img = cv2.imread(os.path.join(orig_image_dir, fname[:5] + "_image.png"))
            if pad_orig_img:
                orig_img = pad_image(orig_img)
            orig_img = cv2.resize(orig_img, (input_wh, input_wh))
            orig_img = orig_img[..., ::-1]
            orig_img = orig_img * (1.0 / 255)

            start = time.time()
            verts = verts_model.predict(input_seg)
            verts_pred_times.append(time.time()-start)

            projects = projects_model.predict(input_seg)

            start = time.time()
            segs = segs_model.predict(input_seg)
            segs_pred_times.append(time.time() - start)
            seg_maps = np.argmax(segs, axis=-1)

            visualise_and_save(fname, orig_img, verts, projects, seg_maps, renderer, input_wh,
                               output_wh, save=save, save_dir=save_dir,
                               overlay_projects=overlay_projects)
            print("Average vertices predict time:", np.mean(verts_pred_times))
            print("Average segmentation predict time:", np.mean(segs_pred_times))


predict_autoencoder(256,
                    48,
                    32,
                    'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_690.hdf5',
                    save=True,
                    save_dir='./results/my_singleperson_imgs/legs_ambiguous/autoencoder_fpn48x48',
                    pad_orig_img=True)
