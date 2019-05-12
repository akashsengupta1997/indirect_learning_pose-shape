import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project, \
    orthographic_project
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.set_cam_params import set_cam_params
from keras_smpl.compute_mask import compute_mask
from renderer import SMPLRenderer
from matplotlib import pyplot as plt

import numpy as np
import deepdish as dd
import pickle

num_vertices = 6890
num_smpls = 1
batch_size = 1
num_camera_params = 4
num_smpl_params = 10 + 72
num_total_params = num_camera_params + num_smpl_params
img_wh = 48
vertex_sampling = 5

random_projects_with_depth_raw = (np.random.rand(num_smpls, num_vertices, 3)).astype('float32')*80

random_projects_with_depth_model = Input(shape=(num_vertices, 3))
mask = Lambda(compute_mask, name="compute_mask")(random_projects_with_depth_model)
segs_model = Lambda(projects_to_seg,
                    arguments={"img_wh": img_wh,
                               "vertex_sampling": vertex_sampling},
                    name="segmentation")([random_projects_with_depth_model, mask])

model2profile = Model(inputs=random_projects_with_depth_model, outputs=segs_model)

# TF Profiling
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
model2profile.compile('adam', 'categorical_crossentropy', run_metadata=run_metadata, options=run_options)

for trial in range(100):
    segs = model2profile.predict(random_projects_with_depth_raw, batch_size=batch_size, verbose=1)

    from tensorflow.python.client import timeline
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    with open('timeline_renderer96_{trial}.json'.format(trial=trial), 'w') as f:
        f.write(trace.generate_chrome_trace_format())
