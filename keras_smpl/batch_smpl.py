"""
Implementation of SMPL function in Keras.
Uses https://github.com/akanazawa/hmr/blob/master/src/tf_smpl/batch_smpl.py as a template.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPLLayer(Layer):
    def __init__(self, pkl_path, batch_size=8, dtype='float32', **kwargs):
        self.pkl_path = pkl_path
        self.dtype = dtype
        self.batch_size = batch_size
        super(SMPLLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Load SMPL params
        with open(self.pkl_path, 'r') as f:
            dd = pickle.load(f)

        # Mean template vertices
        self.v_template = K.variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=self.dtype)

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        # Number of shape parameters
        self.num_betas = dd['shapedirs'].shape[-1]

        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = K.variable(
            shapedir,
            name='shapedirs',
            dtype=self.dtype)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = K.variable(
            dd['J_regressor'].T.todense(),
            name="J_regressor",
            dtype=self.dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207 then transposed
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = K.variable(
            posedirs, name='posedirs', dtype=self.dtype)

        # Indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)
        self.num_joints = self.parents.shape[-1]
        self.num_thetas = self.num_joints * 3

        # LBS weights - i.e. how much the rotation of each joint affects each vertex
        self.lbs_weights = K.variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=self.dtype)

        # # Number of camera parameters
        # self.num_cam = 5

        self.non_trainable_weights = [self.v_template, self.shapedirs, self.J_regressor,
                                      self.posedirs, self.lbs_weights]
        # Add these to trainable weights if want to train them

    def call(self, x):

        thetas = x[:, :self.num_thetas]
        betas = x[:, self.num_thetas:]
        print(tf.shape(x)[0])
        batch_size = tf.shape(x)[0]

        # thetas = x[:, self.num_cam:(self.num_thetas+self.num_cam)]
        # betas = x[:, (self.num_cam + self.num_thetas):]

        # 1. Add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890*3 => N x 6890 x 3
        v_shaped = K.reshape(
            K.dot(betas, self.shapedirs),
            [-1, self.size[0], self.size[1]]) + self.v_template

        # 2. Infer shape-dependent joint locations.
        # (N x 6890) x (6890 x 24) for each x, y, z
        Jx = K.dot(v_shaped[:, :, 0], self.J_regressor)
        Jy = K.dot(v_shaped[:, :, 1], self.J_regressor)
        Jz = K.dot(v_shaped[:, :, 2], self.J_regressor)
        J = K.stack([Jx, Jy, Jz], axis=2)

        # 3. Compute Rodigrues matrices from joint axis angle rotations
        # N x 24 x 3 x 3
        Rs = K.reshape(
            self.batch_rodrigues(K.reshape(thetas, [-1, 3]), batch_size=self.batch_size), [-1, 24, 3, 3])
        # Ignore global rotation.
        pose_feature = K.reshape(Rs[:, 1:, :, :] - K.eye(3), [-1, 207])

        # 4. Add pose blend shapes
        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = K.reshape(
            K.dot(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        # 5. Get the global joint location
        self.J_transformed, A = self.batch_global_rigid_transformation(Rs, J, self.parents)

        # 5. Do skinning:
        # W is N x 6890 x 24
        W = tf.reshape(
            tf.tile(self.lbs_weights, [self.batch_size, 1]), [self.batch_size, -1, 24])
        # (N x 6890 x 24) x (N x 24 x 16)
        T = tf.reshape(
            tf.matmul(W, tf.reshape(A, [self.batch_size, 24, 16])),
            [self.batch_size, -1, 4, 4])
        v_posed_homo = tf.concat(
            [v_posed, tf.ones([self.batch_size, v_posed.shape[1], 1])], 2)
        v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]
        return verts

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],
                        self.v_template.shape[0].value,
                        self.v_template.shape[1].value)
        return output_shape

    def get_config(self):
        config = {'pkl_path': self.pkl_path,
                  'batch_size': self.batch_size,
                  'dtype': self.dtype}
        base_config = super(SMPLLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def batch_global_rigid_transformation(self, Rs, Js, parent, rotate_base=False):
        """
        Computes absolute joint locations given pose.

        rotate_base: if True, rotates the global rotation by 90 deg in x axis.
        if False, this is the original SMPL coordinate.

        Args:
          Rs: N x 24 x 3 x 3 rotation vector of K joints
          Js: N x 24 x 3, joint locations before posing
          parent: 24 holding the parent id for each index

        Returns
          new_J : `Tensor`: N x 24 x 3 location of absolute joints
          A     : `Tensor`: N x 24 x 4 x 4 relative joint transformations for LBS.
        """
        N = Rs.shape[0].value
        if rotate_base:
            print('Flipping the SMPL coordinate frame!!!!')
            rot_x = tf.constant(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
            rot_x = tf.reshape(tf.tile(rot_x, [N, 1]), [N, 3, 3])
            root_rotation = tf.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]

        # Now Js is N x 24 x 3 x 1
        Js = tf.expand_dims(Js, -1)

        def make_A(R, t, name=None):
            # Rs is N x 3 x 3, ts is N x 3 x 1
            with tf.name_scope(name, "Make_A", [R, t]):
                R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])
                t_homo = tf.concat([t, tf.ones([N, 1, 1])], 1)
                return tf.concat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = tf.matmul(
                results[parent[i]], A_here, name="propA%d" % i)
            results.append(res_here)

        # 10 x 24 x 4 x 4
        results = tf.stack(results, axis=1)

        new_J = results[:, :, :3, 3]

        # --- Compute relative A: Skinning is based on
        # how much the bone moved (not the final location of the bone)
        # but (final_bone - init_bone)
        # ---
        Js_w0 = tf.concat([Js, tf.zeros([N, 24, 1, 1])], 2)
        init_bone = tf.matmul(results, Js_w0)
        # Append empty 4 x 3:
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone

        return new_J, A

    def batch_skew(self, vec, input_size=None):
        """
        vec is N x 3, batch_size is int

        returns N x 3 x 3. Skew_sym version of each matrix.
        """
        if input_size is None:
            input_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, input_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [input_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [input_size, 3, 3])

        return res

    def batch_rodrigues(self, theta, batch_size=None):
        """
        Theta is N*K x 3 (reshaped from N*K*3 x 1 before inputting),
        where N is batch_size, K is num_joints = 24

        Output rodrigues angle matrices are N*K x 3 x 3.

        """
        input_size = batch_size * self.num_joints

        angle = K.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = K.expand_dims(tf.div(theta, angle), -1)

        angle = K.expand_dims(angle, -1)
        cos = K.cos(angle)
        sin = K.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")
        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [input_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * self.batch_skew(
            r, input_size=input_size)
        return R





