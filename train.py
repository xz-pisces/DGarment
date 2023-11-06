
import os
import sys
import numpy as np
import tensorflow as tf

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.PBNS import PBNS
from Losses import *
# use 'import' below for parallelized collision loss (specially for bigger batches)
# from LossesRay import *
import cv2
from util import *
from IO import writePC2Frames

from render.render_layer import *

from perceptual.vgg import  LossNetwork



gender = 'M'
""" PARSE ARGS """
gpu_id, name, object, body, checkpoint = parse_args()
if checkpoint is not None:
	checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint

normal_path = './person/'+name+'/normal/'
pgn_path ='./person/'+name+'/all/'
""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100 # update stdout every N steps
if name == 'test': stdout_steps = 1

""" TRAIN PARAMS """
batch_size = 8
num_epochs = 5 if checkpoint is None else 10000

""" SIMULATION PARAMETERS """
edge_young = 15
snug_weight = 1
bend_weight = 5 * 1e-5
collision_weight = 25
collision_dist = .004
mass = .3 # fabric mass as surface density (kg / m2)
blendweights = True# optimize blendweights?

""" MODEL """
print("Building model...")
model = PBNS(name=name,object=object, body=body, checkpoint=checkpoint, blendweights=blendweights)
tgts = model.gather() # model weights
tgts = [v for v in tgts if v.trainable]
model_summary(tgts)
optimizer = tf.optimizers.Adam()

""""summery"""
log_dir = "./logs/" + name
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
summary_writer = tf.summary.create_file_writer(log_dir)


""" DATA """
print("Reading data...")
tr_poses = './person/'+name+'/'+name+'.npy'
tr_data = Data(tr_poses, model._shape, gender, normal_path,pgn_path,batch_size=batch_size)
# tgts = tgts+ [tr_data._poses ]
img_size1 =512
img_size2 =512
tr_steps = floor(tr_data._n_samples / batch_size)
loss_network = LossNetwork()

for epoch in range(num_epochs):
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	step = 0
	metrics = [0] * 4 # Edge, Bend, Gravity, Collisions
	start = time()
	for poses,G, body,dis ,pifuhd,maskhd   in tr_data._iterator:
		""" Train step """

		with tf.GradientTape() as tape:
			pred = model(poses, G)

			L_snug = snug_loss(pred, model.f_connectivity, model.f_connectivity_edges, model.f_area, model._F)
			L_edge, E_edge = edge_loss(pred, model._edges, model._E, weights=model._config['edge'])
			if bend_weight:
				L_bend, E_bend = bend_loss(pred, model._F, model._neigh_F, weights=model._config['bend'])
			else:
				L_bend, E_bend = 0, tf.constant(0)
			# collision
			L_collision, E_collision = collision_loss(pred, model._F, body, model._body_faces, model._config['layers'], thr=collision_dist)
			# gravity
			L_gravity, E_gravity = gravity_loss(pred, surface=model._total_area, m=mass)
			# pin
			if 'pin' in model._config:
				L_pin = tf.reduce_sum(tf.gather(model.D, model._config['pin'], axis=1) ** 2)
			else:
				L_pin = tf.constant(0.0)

			renderer = RenderLayer(img_size1, img_size2, 1, np.ones((6890, 1)), np.zeros(1), model._F ,poses[:,72:76],
								   (img_size1,img_size2),(img_size1/2,img_size2/2),
								   )

			# v = pred
			v1 = pred + dis
			a = v1[:, :, 0]
			b = v1[:, :, 1]*-1
			c = v1[:, :, 2]*-1
			a = tf.expand_dims(a,axis=-1)
			b = tf.expand_dims(b, axis=-1)
			c = tf.expand_dims(c, axis=-1)
			abc = tf.concat([a, b, c], axis=-1)
			rendered_normal,mask = renderer.forward(abc)
			rendered_normal = (rendered_normal+1)/2


			##### perceptual loss
			pifuhd_per = loss_network.forward(pifuhd)
			garment_per = loss_network.forward(rendered_normal)
			normal_loss = loss_network.style_loss(pifuhd_per,garment_per)
			mask_loss = tf.reduce_mean((mask - maskhd) ** 2)

			#
			#
			loss = bend_weight * L_bend * 2000 + \
			   edge_young * L_edge * 50 + \
			   snug_weight * L_snug  + \
			   L_collision* 100 + 50 +\
			   normal_loss * 750 + \
			   mask_loss * 200


		""" Backprop """
		grads = tape.gradient(loss, tgts)		
		optimizer.apply_gradients(zip(grads, tgts))

		

		print("**********************")
		print(step+1)
		print("**********************")

		step += 1

	""" Save checkpoint """
	if (epoch+1)%5==0:
		if not os.path.exists('checkpoints/' + name):
			os.mkdir('checkpoints/' + name)
		model.save('checkpoints/' + name+'/'+str(epoch+1))