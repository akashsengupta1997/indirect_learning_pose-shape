import os

from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
import cv2
import pickle

import tensorflow as tf
from keras.models import load_model

from model import build_full_model_for_predict
from renderer import SMPLRenderer
from preprocessing import pad_image


def realtime_demo(input_wh, output_wh, model_fname, render_vertices=False, segment=False):
    """
    Real-time predictions using auto-encoder model and prepended enet segmentation network.
    :param input_wh:
    :param output_wh:
    :param model_fname:
    :param render_vertices: True = render vertices using opendr (slow). False = just show
    overlayed 2D projections of vertices.
    :param segment: segment vertices into body-parts
    """
    renderer = SMPLRenderer()
    smpl_model = load_model(os.path.join("./autoencoder_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Autoencoder SMPL Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model_for_predict(smpl_model,
                                                                           output_wh,
                                                                           "./neutral_smpl_with_cocoplus_reg.pkl")

    enet_model_path = './enet_weights/enet256_small_glob_rot_no_horiz_flip0401.hdf5'
    enet_model = load_model(enet_model_path)
    print('ENet Model {enet_model_path} loaded'.format(enet_model_path=enet_model_path))

    with open("./keras_smpl/part_vertices.pkl", 'rb') as f:
        part_indices = pickle.load(f)

    projects_colour_map = [0]*6890
    i = 0
    for part in part_indices:
        for index in part:
            projects_colour_map[index] = float(i)/31
        i += 1

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, orig_img = cap.read()
        orig_width = orig_img.shape[1]
        crop_img = orig_img[:, int(0.25*orig_width):int(0.75*orig_width), :]
        pad_img = pad_image(crop_img)
        bgr_img = cv2.resize(pad_img, (input_wh, input_wh))
        img = bgr_img[..., ::-1]
        img = img * (1/255.0)
        # Add batch dimension: 1 x D x D x 3
        img_tensor = np.expand_dims(img, 0)
        seg_tensor = np.reshape(enet_model.predict(img_tensor),
                                (-1, input_wh, input_wh, 32))
        seg_img_tensor = np.argmax(seg_tensor, axis=-1)
        seg_img_tensor = np.expand_dims(seg_img_tensor, axis=-1)
        seg_img_tensor = seg_img_tensor * (1.0 / 31)

        if render_vertices:
            # RENDERING VERTICES IS SLOW
            verts = verts_model.predict(seg_img_tensor)
            rend_img = renderer(verts=verts[0], render_seg=segment)

            # Display
            display_img = cv2.flip(cv2.resize(bgr_img, (512, 512)), 1)
            cv2.namedWindow("img")
            cv2.imshow('img', display_img)
            cv2.namedWindow("rend", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('rend', 600, 600)
            cv2.imshow('rend', rend_img)

        else:
            # PLOTTING PROJECTS IS FASTER
            projects = projects_model.predict(seg_img_tensor)
            plt.figure(1,figsize=(10,10))
            plt.clf()
            scatter_scale = float(input_wh) / output_wh
            if segment:
                plt.scatter(projects[0, :, 0] * scatter_scale,
                            projects[0, :, 1] * scatter_scale,
                            s=1,
                            c=projects_colour_map)
            else:
                plt.scatter(projects[0, :, 0] * scatter_scale,
                            projects[0, :, 1] * scatter_scale,
                            s=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.imshow(cv2.flip(img, 0),
                       alpha=0.9)
            plt.gca().invert_yaxis()
            plt.draw()
            plt.pause(0.0001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


realtime_demo(256,
              64,
              'up-s31_64x64_resnet_ief_scaledown0005_arms_weighted2_bg_weighted_0point3_gamma2_600.hdf5',
              render_vertices=False,
              segment=True)
