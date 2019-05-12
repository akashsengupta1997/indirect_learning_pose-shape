from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D, \
    BatchNormalization, Activation, Add, Concatenate
from keras.applications import resnet50

from keras_smpl.batch_smpl import SMPLLayer
from keras_smpl.projection import persepective_project, orthographic_project
from keras_smpl.projects_to_seg import projects_to_seg
from keras_smpl.concat_mean_param import concat_mean_param
from keras_smpl.set_cam_params import load_mean_set_cam_params
from keras_smpl.compute_mask import compute_mask

from encoders.encoder_enet_simple import build_enet


def build_model(train_batch_size, input_shape, smpl_path, output_wh, num_classes,
                encoder_architecture='resnet50', use_IEF=False, vertex_sampling=None,
                scaledown=0.005):
    num_camera_params = 4
    num_smpl_params = 72 + 10
    num_total_params = num_smpl_params + num_camera_params

    # --- BACKBONE ---
    if encoder_architecture == 'enet':
        inp = Input(shape=input_shape)
        img_features = build_enet(inp)  # (N, 32, 32, 128) output size from enet

    elif encoder_architecture == 'resnet50':
        resnet = resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape)
        inp = resnet.input
        img_features = resnet.output
        img_features = Reshape((2048,))(img_features)

    if use_IEF:
        # --- IEF MODULE ---
        # Instantiate ief layers
        IEF_layer_1 = Dense(1024, activation='relu', name='IEF_layer_1')
        IEF_layer_2 = Dense(1024, activation='relu', name='IEF_layer_2')
        IEF_layer_3 = Dense(num_total_params, activation='linear', name='IEF_layer_3')

        state1 = Lambda(concat_mean_param,
                        arguments={'img_wh': output_wh})(img_features)
        param1 = Lambda(lambda x: x[:, 2048:])(state1)
        print("State1 shape", state1.get_shape())
        print("Param1 shape", param1.get_shape())

        # Iteration 1
        delta1 = IEF_layer_1(state1)
        delta1 = IEF_layer_2(delta1)
        delta1 = IEF_layer_3(delta1)
        delta1 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta1)
        param2 = Add()([param1, delta1])
        state2 = Concatenate()([img_features, param2])

        # Iteration 2
        delta2 = IEF_layer_1(state2)
        delta2 = IEF_layer_2(delta2)
        delta2 = IEF_layer_3(delta2)
        delta2 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta2)
        param3 = Add()([param2, delta2])
        state3 = Concatenate()([img_features, param3])

        # Iteration 3
        delta3 = IEF_layer_1(state3)
        delta3 = IEF_layer_2(delta3)
        delta3 = IEF_layer_3(delta3)
        delta3 = Lambda(lambda x, d: x * d, arguments={"d": scaledown})(delta3)
        final_param = Add()([param3, delta3])

    else:
        smpl = Dense(2048, activation='relu')(img_features)
        smpl = Dense(1024, activation='relu')(smpl)
        smpl = Dense(num_total_params, activation='linear')(smpl)
        smpl = Lambda(lambda x: x * scaledown, name="scale_down")(smpl)
        final_param = Lambda(load_mean_set_cam_params,
                             arguments={'img_wh': output_wh})(smpl)

    verts = SMPLLayer(smpl_path, batch_size=train_batch_size)(final_param)
    projects_with_depth = Lambda(orthographic_project,
                                 arguments={'vertex_sampling': vertex_sampling},
                                 name='project')([verts, final_param])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': vertex_sampling},
                  name='segment')([projects_with_depth, masks])
    segs = Reshape((output_wh * output_wh, num_classes), name="final_reshape")(segs)
    segs = Activation('softmax', name="final_softmax")(segs)

    segs_model = Model(inputs=inp, outputs=segs)
    smpl_model = Model(inputs=inp, outputs=final_param)
    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)

    print(segs_model.summary())

    return segs_model, smpl_model, verts_model, projects_model


def build_full_model_from_saved_model(smpl_model, output_wh, smpl_path, batch_size,
                                      num_classes):
    inp = smpl_model.input
    smpl = smpl_model.output
    verts = SMPLLayer(smpl_path, batch_size=batch_size)(smpl)
    projects_with_depth = Lambda(orthographic_project,
                                 arguments={'vertex_sampling': None},
                                 name='project')([verts, smpl])
    masks = Lambda(compute_mask, name='compute_mask')(projects_with_depth)
    segs = Lambda(projects_to_seg,
                  arguments={'img_wh': output_wh,
                             'vertex_sampling': None},
                  name='segment')([projects_with_depth, masks])
    segs = Reshape((output_wh * output_wh, num_classes), name="final_reshape")(segs)
    segs = Activation('softmax', name="final_softmax")(segs)

    verts_model = Model(inputs=inp, outputs=verts)
    projects_model = Model(inputs=inp, outputs=projects_with_depth)
    segs_model = Model(inputs=inp, outputs=segs)

    return verts_model, projects_model, segs_model