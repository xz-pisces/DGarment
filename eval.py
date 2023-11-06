import os
import sys
import numpy as np
import tensorflow as tf
from math import floor

from Data.data import Data
from Model.PBNS import PBNS
import time
from util import parse_args, model_summary
from IO import writeOBJ, writePC2Frames

"""
This script will load a PBNS checkpoint
Predict results for a set of poses (at 'Data/test.npy')
Store output animation data in 'results/' folder
	Body:
	- 'results/body.obj'
	- 'results/body.pc2'
	Outfit:
	- 'results/outfit.obj'
	- 'results/ouftit.pc2'
	- 'results/rest.pc2' -> before skinning
"""

""" PARSE ARGS """
localtime = time.time()
time = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))
# print(time)
gender = 'M'
gpu_id, name, object, body, checkpoint = parse_args(train=False)
namefile = os.path.abspath(os.path.dirname(__file__)) + '/results/' + name+'_'+time + '/'
if not os.path.isdir(namefile):
	os.mkdir(namefile)
checkpoint = './checkpoints/'+name+'/'+checkpoint



normal_path = './person/'+name+'/normal/'
pgn_path ='./person/'+name+'/all/'
""" PARAMS """
batch_size = 1

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" MODEL """
print("Building model...")
model = PBNS(name=name ,object=object, body=body, checkpoint=checkpoint)
tgts = model.gather() # model weights
model_summary(tgts)

""" DATA """
print("Reading data...")
val_poses = './person/'+name+'/'+name+'.npy'
val_data = Data(val_poses, model._shape,gender,normal_path,pgn_path, batch_size=batch_size, mode='test')
val_steps = floor(val_data._n_samples / batch_size)

""" CREATE BODY AND OUTFIT .OBJ FILES """
writeOBJ(namefile + 'body.obj', model._body, model._body_faces)
writeOBJ(namefile + 'outfit.obj', model._T, model._F)

""" EVALUATION """
print("")
print("Evaluating...")
print("--------------------------")
step = 0

#
for poses,G, body,dis ,pifuhd,maskhd   in val_data._iterator:
	pred = model(poses, G)
	if not os.path.exists(namefile+'/garments_wo/'):
		os.mkdir(namefile+'/garments_wo/')
	writeOBJ(namefile+'/garments_wo/'+str(step)+'garment.obj' ,pred[0].numpy(), model._F)
	sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(val_steps))
	sys.stdout.flush()
	step += 1
print("")
print("")
print("DONE!")
print("")