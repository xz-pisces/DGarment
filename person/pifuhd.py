import torch
import cv2
import numpy as np
import os
torch.backends.cudnn.benchmark = True

BBB = torch.jit.load("netB.pth").to("cuda:0")


imgsize = 1080
name = 'male-3-sport'
path = './'+name+'/imgs/'
# path = '/home/lxz/mygrade4/RobustVideoMatting-master/tem/iper016_2/'
pgnpath = './'+name+'/all/'
results = './'+name+'/normal/'

upperpath = './'+name+'/upper/'
upperresults = './'+name+'/uppernormal/'
downpath = './'+name+'/down/'
downresults = './'+name+'/downnormal/'

files = sorted(os.listdir(path))
print(len(files))
if not os.path.exists(results):
    os.makedirs(results)
if not os.path.exists(upperresults):
    os.makedirs(upperresults)
if not os.path.exists(downresults):
    os.makedirs(downresults)

# print(files)
for i in files:
    FFF = torch.jit.load("netF.pth").to("cuda:0")
    print(i)
    a = cv2.imread(path+i)[:,:,::-1]
    a = a.astype(np.float32)
    a = torch.from_numpy(a).to("cuda:0")/127.5-1

    a = a.permute(2,0,1)[None]
    
    upperpng = cv2.imread(upperpath+i)[:,:,::-1]
    downpng = cv2.imread(downpath+i)[:,:,::-1]
    png = cv2.imread(pgnpath+i)[:,:,::-1]
    with torch.no_grad():
        b = FFF(a)
        # c = BBB(a)

        b = b[0].permute(1,2,0).cpu().numpy()*127.5+127.5
        b = cv2.resize(b, (imgsize,imgsize), interpolation=cv2.INTER_AREA) 
        print(b.shape)
        imgs = b*png/255

        upperimgs = b*upperpng/255
        downimgs = b*downpng/255

        cv2.imwrite(upperresults+i, upperimgs[:,:,::-1])
        cv2.imwrite(downresults+i, downimgs[:,:,::-1])

        cv2.imwrite(results+i, imgs[:,:,::-1])

