import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry as tfg
from tensorflow_graphics.geometry import representation
# from tensor
# import tensorflow_graphics as tfg
from scipy.spatial import cKDTree
from snug.layers import *

def edge_loss(x, y_e, E, weights):
	x_e = tf.gather(x, E[:,0], axis=1) - tf.gather(x, E[:,1], axis=1)
	x_e = tf.sqrt(tf.reduce_sum(x_e ** 2, -1))
	d_e = (x_e - y_e[None])
	err = tf.reduce_mean(tf.abs(d_e))
	d_e = d_e ** 2
	d_e = tf.multiply(weights[None], d_e)
	loss = tf.reduce_sum(d_e)
	return loss, err

def snug_loss(v,F,E,A,faces):
	v = tf.convert_to_tensor(v, tf.float32)
	faces = tf.convert_to_tensor(faces, tf.int32)
	fn = FaceNormals(dtype=v.dtype)(v, faces)
	n0 = tf.gather(fn, F[:, 0], axis=1)
	n1 = tf.gather(fn, F[:, 1], axis=1)

	# Compute edge lenght
	v0 = tf.gather(v, E[:, 0], axis=1)
	v1 = tf.gather(v, E[:, 1], axis=1)
	e = v1 - v0
	e_norm, l = tf.linalg.normalize(e, axis=-1)

	# Compute area
	f_area = tf.repeat([A], tf.shape(v)[0], axis=0)
	a0 = tf.gather(f_area, F[:, 0], axis=1)
	a1 = tf.gather(f_area, F[:, 1], axis=1)
	a = a0 + a1

	# Compute dihedral angle between faces
	cos = tf.reduce_sum(tf.multiply(n0, n1), axis=-1)
	sin = tf.reduce_sum(tf.multiply(e_norm, tf.linalg.cross(n0, n1)), axis=-1)
	theta = tf.math.atan2(sin, cos)
	# theta = tf.math.acos(cos)

	# Compute bending coefficient according to material parameters,
	# triangle areas (a) and edge length (l)

	scale = l[..., 0] ** 2 / (4 * a)

	# Bending energy
	energy =  scale * (theta ** 2) / 2


	return tf.reduce_sum(energy)

def bend_loss(x, F, neighF, weights):
	# compute face normals
	V02 = tf.gather(x, F[:,2], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
	V01 = tf.gather(x, F[:,1], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
	N = tf.linalg.cross(V02, V01) # B x N x 3
	N = N / tf.linalg.norm(N, axis=-1, keepdims=True) # B x N x 3
	# compare neighbouring face normals
	N0 = tf.gather(N, neighF[:,0], axis=1) # B x N x 3
	N1 = tf.gather(N, neighF[:,1], axis=1) # B x N x 3
	# cosinus distances
	cos = tf.reduce_sum(tf.multiply(N0, N1), -1)
	err = tf.reduce_mean(1 - cos)
	dN = (N0 - N1) ** 2
	dN = tf.multiply(weights[None,:,None], dN)
	loss = tf.reduce_sum(dN)
	return loss, err

""" COLLISION FUNCTIONS START """
def collision_loss(V, F, B, B_F, layers, thr=.004, stop_gradient=False):
	V_vn =representation.mesh.normals.vertex_normals(V, tf.tile(F[None], [V.shape[0], 1, 1]))
	B_vn = representation.mesh.normals.vertex_normals(B, tf.tile(B_F[None], [B.shape[0], 1, 1]))
	loss = 0
	vcount = np.array([0] * len(layers), np.float32)
	for i in range(len(layers)):
		l = layers[i]
		# get layer verts
		v = tf.gather(V, l, axis=1)
		# compute correspondences
		idx = _tf_nn_parallel(v, B)
		N, M = v.shape[:2]
		_idx = tf.tile(np.array(list(range(N)), np.int32)[:,None], [1, M])
		idx = tf.stack((_idx, idx), axis=-1)
		# compute loss
		D = v - tf.gather_nd(B, idx)
		b_vn = tf.gather_nd(B_vn, idx)
		dot = tf.einsum('abc,abc->ab', D, b_vn)
		loss += tf.reduce_sum(tf.minimum(dot - thr, 0) ** 2)
		# vcount
		vmask = tf.cast(tf.math.less(dot, 0), tf.float32)
		vcount[i] = tf.reduce_sum(vmask).numpy() / (N * M)
		# add layer to "body"
		if i + 1 < len(layers):
			if stop_gradient:
				v = tf.stop_gradient(v)
				v_vn = tf.stop_gradient(tf.gather(V_vn, l, axis=1))
			else:
				v_vn = tf.gather(V_vn, l, axis=1)
			B = tf.concat((B, v), axis=1)
			B_vn = tf.concat((B_vn, v_vn), axis=1)
	return loss, vcount

def _nearest_neighbour(V, B):
	tree = cKDTree(B.numpy())
	return tree.query(V.numpy(), n_jobs=-1)[1]

@tf.function
def _tf_nn_parallel(V, B):
	return tf.map_fn(fn=lambda elem: tf.py_function(func=_nearest_neighbour, inp=elem, Tout=tf.int32), elems=(V,B), dtype=tf.int32, parallel_iterations=V.shape[0])
""" COLLISION FUNCTIONS END """

def gravity_loss(x, surface, m=.15, g=9.81):
	vertex_mass = surface * m / x.shape[1]
	U = vertex_mass * g * x[:,:,2]
	return tf.reduce_sum(U), tf.reduce_mean(tf.reduce_sum(U, -1))