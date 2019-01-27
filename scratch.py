import pickle
import numpy as np
import tensorflow as tf

from renderer import SMPLRenderer
from matplotlib import pyplot as plt
from tf_smpl.batch_smpl import undo_chumpy, SMPL

smpl_path = "./neutral_smpl_with_cocoplus_reg.pkl"

# Messing  around with rendering
# with open(smpl_path, 'r') as f:
#     smpl = pickle.load(f)
#
# print(smpl.keys())
#
# mean_template_vertices = undo_chumpy(smpl['v_template'])
# num_vertices = mean_template_vertices.shape[0]
# print(mean_template_vertices.shape)
#
# renderer = SMPLRenderer()
#
# mean_render = renderer(verts=mean_template_vertices)
# plt.figure(1)
# plt.clf()
# plt.imshow(mean_render)
#
# num_beta = 10
# shapedirs = np.reshape(undo_chumpy(smpl['shapedirs']), [-1, num_beta]).T
# print(shapedirs.shape)
#
# beta = np.zeros((2, num_beta))
# beta[0][0] = 2
# beta[1][1] = 2
# v_shaped = np.reshape(np.dot(beta, shapedirs), [-1, num_vertices, 3]) + mean_template_vertices
# print(v_shaped.shape)
# v_shaped_render_0 = renderer(verts=v_shaped[0])
# plt.figure(2)
# plt.clf()
# plt.imshow(v_shaped_render_0)
# v_shaped_render_1 = renderer(verts=v_shaped[1])
# plt.figure(3)
# plt.clf()
# plt.imshow(v_shaped_render_1)
# plt.show()


# Playing with the given SMPL function
num_beta = 10
num_theta = 72
dtype = np.float32
batch_size = 3
smpl = SMPL(pkl_path=smpl_path)

beta = np.zeros((batch_size, num_beta))
beta[0][0] = 2
beta[1][1] = 2
beta = tf.Variable(beta, name='beta', dtype=dtype)

theta = np.zeros((batch_size, num_theta))
theta = tf.Variable(theta, name='theta', dtype=dtype)

verts, _, _ = smpl(beta, theta, get_skin=True)
print(verts.shape)
print(type(verts))

renderer = SMPLRenderer()
rend_img0 = renderer(verts=verts[0])
rend_img1 = renderer(verts=verts[1])
rend_img2 = renderer(verts=verts[2])
plt.figure()
plt.subplot(311)
plt.imshow(rend_img0)
plt.subplot(312)
plt.imshow(rend_img1)
plt.subplot(313)
plt.imshow(rend_img2)
plt.show()

asdasd