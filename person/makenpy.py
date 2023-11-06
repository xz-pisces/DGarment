# coding=utf-8
import h5py
import numpy as np
import pickle
import joblib

with open('./zl2/output.pkl', 'rb') as f:
    lll = joblib.load(f)

data = lll[1]



a = data['pose']
print(a.shape)
b = data['orig_cam']
c = data['betas']
d = np.arange(0,a.shape[0])
print(d)
d=np.expand_dims(d, axis=-1) 


final = np.concatenate((a,b,c,d),axis=-1)
print(final.shape)
np.save('./zl2/zl2.npy',final)


