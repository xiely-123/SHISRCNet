import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import time
import random
import glob
import math
import cv2
import core 

def bicubic(x, ratio=2.0):
    # If the input is on a CUDA device, all computations will be done using the GPU.
    x = torch.Tensor(x)
    x_resized = core.imresize(x, scale=ratio, antialiasing=False)
    return x_resized.numpy()


# Larger batch sizes are also supported.

def image_to_numpy(img):
    img_io = imageio.imread(img)
    img_np = np.array(img_io)
    np_img = np.ascontiguousarray(img_np.transpose((2, 0, 1)))
    return np_img
    
def data_read(path): 
    #print(path.shape)
    HR = cv2.imread(path[0].replace('histology_slides_resize','histology_slides_resize'))
    LR = cv2.imread(path[0].replace('histology_slides_resize','histology_slides_x2')) #x2,x4,LR
    HR = HR/255
    LR = LR/255
    batchLR = LR.T   
    batchHR = HR.T
    label = int(path[1])
    return np.array(batchLR), np.array(batchHR), np.array(label)

def data_read_mask(path): 
    #print(path.shape)
    HR = cv2.imread(path.replace('histology_slides_resize','histology_slides_resize'))
    LR = cv2.imread(path.replace('histology_slides_resize','histology_slides_LR'))
    batchLR_8 = bicubic(LR.T,4)/255
    HR = HR/255
    LR = LR/255
    batchLR = LR.T   
    batchHR = HR.T
    return np.array(batchLR), np.array(batchHR), np.array(batchLR_8) 

class SPDataSet(Data.Dataset):
    def __init__(self, path):
        super(SPDataSet,self).__init__()
        self.path = path[:] ######path
    def __len__(self):
        return len(self.path)
 
 
    def __getitem__(self, index):
 
        '''load the datas'''
        LR, HR, label = data_read(self.path[index])
        return LR, HR, label



class SPDataSet_mask(Data.Dataset):
    def __init__(self, path):
        super(SPDataSet_mask,self).__init__()
        self.path = path[:] ######path
    def __len__(self):
        return len(self.path)
 
 
    def __getitem__(self, index):
 
        '''load the datas'''
        LR, HR, LR_8 = data_read_mask(self.path[index])
        return LR, HR, LR_8



"""
class DataLoaderX(Data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

start_time = time.time()
dim = 32000

data = np.loadtxt("./HR_groudtruth.txt", dtype=str, delimiter=',')
path = np.loadtxt("./HR_groudtruth.txt", dtype=str)


#print(feature.shape, clean.shape, MIX.shape)

torch_dataset = SPDataSet(path)
loader = Data.DataLoader(dataset = torch_dataset, batch_size = 16, shuffle = False,num_workers = 4,)
for i in range(1):
    for a,b in loader:
        print(i,a.shape)
        print(time.time()-start_time)
"""