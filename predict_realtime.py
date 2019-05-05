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


def realtime_demo(input_wh, output_wh, model_fname):
    renderer = SMPLRenderer()
    smpl_model = load_model(os.path.join("./full_network_weights", model_fname),
                            custom_objects={'dd': dd,
                                            'tf': tf})
    print('Model {model_fname} loaded'.format(model_fname=model_fname))

    verts_model, projects_model, segs_model = build_full_model(smpl_model,
                                                               output_wh,
                                                               "./neutral_smpl_with_cocoplus_reg.pkl")

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, orig_img = cap.read()
        orig_img = cv2.resize(orig_img, (300, 400))  # extra resize to deal w/ mac aspect ratio
        pad_img = pad_image(orig_img)
        bgr_img = cv2.resize(pad_img, (input_wh, input_wh))
        img = bgr_img[..., ::-1]
        img = img * (1/255.0)
        # Add batch dimension: 1 x D x D x 3
        img_tensor = np.expand_dims(img, 0)

        verts = verts_model.predict(img_tensor)
        rend_img = renderer(verts=verts[0], render_seg=False)

        # Display
        display_img = cv2.flip(cv2.resize(bgr_img, (512, 512)), 1)
        plt.figure(1)
        plt.imshow(rend_img)
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()
        cv2.imshow('img', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


realtime_demo(256, 64, 'up-s31_48x48_resnet_ief_scaledown0005_arms_weighted_2_bg_weighted_0point3_gamma2_1630.hdf5')