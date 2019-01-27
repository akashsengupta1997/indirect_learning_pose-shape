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
batch_size = 4

smpl = SMPL(pkl_path=smpl_path)

beta = (np.random.randn(batch_size, num_beta) - 0.5) * 0.5
beta = tf.Variable(beta, name='beta', dtype=dtype, trainable=False)

theta = (np.random.rand(batch_size, num_theta) - 0.5) * 0.5
theta = tf.Variable(theta, name='theta', dtype=dtype, trainable=False)

verts, _, _ = smpl(beta, theta, get_skin=True)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
result = sess.run({'verts': verts})
print(result['verts'].shape)
print(type(result['verts']))

result_verts = result['verts']
renderer = SMPLRenderer()
rend_img0 = renderer(verts=result_verts[0])
rend_img1 = renderer(verts=result_verts[1])
rend_img2 = renderer(verts=result_verts[2])
rend_img3 = renderer(verts=result_verts[3])
plt.figure()
plt.subplot(221)
plt.imshow(rend_img0)
plt.subplot(222)
plt.imshow(rend_img1)
plt.subplot(223)
plt.imshow(rend_img2)
plt.subplot(224)
plt.imshow(rend_img3)
plt.show()

# PLY file and vertex colouring
# from plyfile import PlyData, PlyElement
# bodypart_ply = "template-bodyparts.ply"
# with open(bodypart_ply, 'rb') as f:
#     plydata = PlyData.read(f)
#     print(plydata.elements)
#     for vertex in plydata.elements[0].data:
#         print('rgb', vertex[-3], vertex[-2], vertex[-1])

