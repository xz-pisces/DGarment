import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smpl.smpl_np import SMPLModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *
import cv2

class Data:	
	def __init__(self, poses, shape, gender,normal,pgn_path, batch_size=10, mode='train'):
		"""
		Args:
		- poses: path to .npy file with poses
		- shape: SMPL shape parameters for the subject
		- gender: 0 = female, 1 = male
		- batch_size: batch size
		- shuffle: shuffle
		"""
		# Read sample list
		self._poses = np.load(poses)
		if self._poses.dtype == np.float64: self._poses = np.float32(self._poses)
		self._poses = tf.Variable(self._poses, name='POSE', trainable=True)
		self._n_samples = self._poses.shape[0]
		# smpl
		smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
		smpl_path += 'model_[G].pkl'.replace('[G]', 'neutral' if gender=='M' else 'neutral')
		self.SMPL = SMPLModel(smpl_path, rest_pose)
		self._shape = shape
		# TF Dataset
		ds = tf.data.Dataset.from_tensor_slices(self._poses)
		if mode == 'train': ds = ds.shuffle(self._n_samples)
		ds = ds.map(self.tf_map, num_parallel_calls=batch_size)
		ds = ds.batch(batch_size=batch_size)
		self._iterator = ds
		self.normal_path = normal
		self.pgn_path =pgn_path
		self.imgfiles = sorted(os.listdir(normal))
		self.pgnfiles = sorted(os.listdir(pgn_path))


	def _next(self, pose):
		# compute body
		# while computing SMPL should be part of PBNS, 
		# if it is in Data, it can be efficiently parallelized without overloading GPU
		# pose = pose[:72]
		# trans
		# print(int(pose[-1]))
		pifuhd =self.imgfiles[int(pose[-1])]
		maskhd = self.pgnfiles[int(pose[-1])]
		hdimg = cv2.imread(self.normal_path+pifuhd, -1)
		maskimg =  cv2.imread(self.pgn_path+maskhd, -1)
		# gtimg = torch.from_numpy(hdimg.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

		hdimg = hdimg.astype(np.float32) / 255.0
		maskimg = maskimg.astype(np.float32) / 255.0
		G, B,dis = self.SMPL.set_params(pose=pose[:72].numpy(), beta=self._shape, with_body=True)
		return pose, G, B,dis,hdimg,maskimg
		
	def tf_map(self, pose):
		return tf.py_function(func=self._next, inp=[pose], Tout=[tf.float32, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32])
