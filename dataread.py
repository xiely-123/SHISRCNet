# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:40:27 2020

@author: ALW
"""


import numpy as np    ##引入numpy模块，主要是处理矩阵的函数，定义为np
import os, random   ##os索引模块（比如索引GPU、路径等），random选择模块（比如选择数据个数等）
from PIL import Image ####Image:引入图像读取函数
from torchvision import transforms as transforms   ####引入torchvision中的transforms，做图像增强操作
import cv2
import core 
import torch
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
    
    
def normalization(data):   ########def定义归一化函数，主要的目的：使模型更快收敛
    #_range = np.max(data) - np.min(data)
    #return (data - np.min(data)) / _range

    _range = np.max(abs(data))      ######找到图像中绝对值的最大值
    return data / _range    #######数据除以最大值，使数据在[-1,1]之间，让模型收敛更快，使训练效果更好


def data_argument_function(data,groudturth,data_resize, degree, angle):  ###定义数据增强函数，影像数据、标签数据、旋转角度（自定义）；3类数据增强，其他增强对数据有损伤，只有以下三类增强比较适用于医学图片分割
    #print(degree)
    if degree ==0:
        data = transforms.RandomHorizontalFlip(p=1)(Image.fromarray(data))   ####水平翻转，目前就这三种增强分割可以使用。
        groudturth = transforms.RandomHorizontalFlip(p=1)(Image.fromarray(groudturth))  #p=1是固定参数
        data_resize = transforms.RandomHorizontalFlip(p=1)(Image.fromarray(data_resize))
    if degree == 1:          
        data = transforms.RandomVerticalFlip(p=1)(Image.fromarray(data))    ####上下翻转，目前就这三种增强分割可以使用
        groudturth = transforms.RandomVerticalFlip(p=1)(Image.fromarray(groudturth))
        data_resize = transforms.RandomVerticalFlip(p=1)(Image.fromarray(data_resize))
    #if degree == 2:          
    #    data = transforms.RandomRotation((angle,angle))(Image.fromarray(data))    ####上下翻转，目前就这三种增强分割可以使用
    #    groudturth = transforms.RandomRotation((angle,angle))(Image.fromarray(groudturth))
    #    data_resize = transforms.RandomRotation((angle,angle))(Image.fromarray(data_resize))    
    else: 
        data = data
        groudturth = groudturth
        data_resize = data_resize
    return np.asarray(data), np.asarray(groudturth), np.asarray(data_resize)    #####回传三类增强后的数据与label





def batch_data_read(path): 
    #print(path.shape)
    batchLR=[]   
    batchHR=[]  
    for i in range(len(path)):
        HR = cv2.imread(path[i].replace('histology_slides_resize','histology_slides_resize'))
        LR = cv2.imread(path[i].replace('histology_slides_resize','histology_slides_LR'))
        #print(HR.shape,LR.shape)
        #img_resize = cv2.resize(img, (496, 368))
        #print(img_resize.shape)
        HR = HR/255
        LR = LR/255
        batchLR.append(LR.T)   
        batchHR.append(HR.T) 
    return np.array(batchLR), np.array(batchHR) 


def batch_data_read_test(path): 
    #print(path.shape)
    batchLR=[]   
    batchHR=[]  
    for i in range(len(path)):
        HR = cv2.imread(path[i].replace('histology_slides_resize','histology_slides_resize'))
        LR = cv2.imread(path[i].replace('histology_slides_resize','histology_slides_LR'))
        HR = HR/255
        LR = LR/255
        batchLR.append(LR.T)   
        batchHR.append(HR.T) 
    return np.array(batchLR), np.array(batchHR) 
"""
data = np.loadtxt("./HR_groudtruth.txt", dtype=str, delimiter=',')
a,b = batch_data_read(data)
print(a.shape,b.shape)
"""
def batch_data_read_AU(path): 
    #print(path.shape)
    data=[]   
    data_resize = []
    groudturth=[]  
    for i in range(len(path)):
        img = cv2.imread(path[i][0])
        img1 = cv2.imread(path[i][1])
        #print(img.shape)
        img_resize = cv2.resize(img, (496, 368))
        #print(img_resize.shape)
        angle = random.randint(-80,80)
        degree = random.randint(0,3)
        img, img1, img_resize = data_argument_function(img, img1, img_resize, degree, angle)
        #print(degree,angle)
        img = img/255
        img_resize = img_resize/255
        img1 = img1/255
        data.append(img.T)   
        groudturth.append(img1.T) 
        data_resize.append(img_resize.T) 
    return np.array(data), np.array(groudturth), np.array(data_resize)  


    
"""
data = np.loadtxt("./data_label-4.txt", dtype=str, delimiter=',')
a,b,c = batch_data_read(data)
print(a.shape,b.shape,c.shape)
"""
def test_data_read(path): 
    print(path.shape)
    batchLR=[]   
    batchHR=[]    
    for i in range(1):
        HR = cv2.imread(path.replace('histology_slides_resize','histology_slides_resize'))
        LR = cv2.imread(path.replace('histology_slides_resize','histology_slides_LR'))
        HR = HR/255
        LR = LR/255
        batchLR.append(LR.T)   
        batchHR.append(HR.T) 
        cv2.imwrite("/home/xly/SR_classification/results/LR.jpg", LR.T.T*255)
        cv2.imwrite("/home/xly/SR_classification/results/HR.jpg", HR.T.T*255)
    return np.array(batchLR), np.array(batchHR) 

def test_data_read_SRCNN(path,ratio): 
    print(path.shape)
    batchLR=[]   
    batchHR=[]    
    for i in range(1):
        HR = cv2.imread(path.replace('histology_slides_resize','histology_slides_resize'))
        LR = cv2.imread(path.replace('histology_slides_resize','histology_slides_x4'))
        HR = HR/255
        #LR = LR/255
        batchLR.append(bicubic(LR.T,ratio)/255)   
        batchHR.append(HR.T) 
        cv2.imwrite("/home/xly/SR_classification/results/LRbicubic.jpg", bicubic(LR.T,ratio).T)
        cv2.imwrite("/home/xly/SR_classification/results/LR.jpg", LR)
        cv2.imwrite("/home/xly/SR_classification/results/HR.jpg", HR.T.T*255)
    return np.array(batchLR), np.array(batchHR) 


"""
data = np.loadtxt("./data_label.txt", dtype=str, delimiter=',')
a,b = test_data_read(data[1])
print(a.shape,b.shape)
"""








'''
readdata主要是通过相应的png文件映射到dcm文件上，进行训练。


def readdata(image_dir,label_dir,batch_size,data_argument=True): ##定义数据读取函数，将归一化和增强函数都加进来了。data_argument是否进行数据增强
        data=[]   ######数据列表初始化
        label=[]  ######label列表初始化
        pathDir = os.listdir(label_dir)   #####使用os.listdir读取label_dir文件夹里的每条数据名称
        #print(pathDir)#取图片的原始路径
        filenumber=len(pathDir)  #######看数据个数
        #print(filenumber)
        sample = random.sample(pathDir, batch_size)  ########在label数据中选择batch_size个数据，赋值给sample
        #print (sample)

        for name in sample:   ####读取选择数据，将sample逐个赋值给name
            if data_argument: #判断是否增强
                label_data = Image.open(label_dir+name)    #######打开label jpg文件
                la_array = np.array(label_data)            ######将JPG数据转为numpy矩阵格式
                name=name.replace('..png','..dcm')         ######将label中的..png格式转为..dcm；name.replace：name是指sample赋值给的name
                image = pydicom.read_file(image_dir+name)  ######使用pydicom.read_file打开dcm文件             
                im_array = image.pixel_array               ######image是dcm数据，读取dcm数据中的矩阵
                im_array = normalization(im_array)         ######数据归一化
                ###增强
                a,a1,a2,b,b1,b2=data_argument_function(im_array,la_array,random.randint(-80,80)) ######采用了数据增强，random.randint选择旋转角度（从-80°到80°之间选择一个角度值）
                data.append(im_array)  ##append是Python自带的拼接函数，将dcm数据矩阵拼接到初始化的data列表中
                data.append(np.reshape(a, [512,512]))  #####数据a拼接到data当中，数据要一个一个拼接
                data.append(np.reshape(a1, [512,512])) #####数据a1拼接到data当中，数据要一个一个拼接
                data.append(np.reshape(a2, [512,512])) #####数据a2拼接到data当中，数据要一个一个拼接
                label.append(la_array)  
                label.append(np.reshape(b, [512,512])) #####label，b拼接到label当中
                label.append(np.reshape(b1, [512,512]))#####label拼接
                label.append(np.reshape(b2, [512,512]))#####label拼接

        
            else:   ######不采用数据增强的时候，数据读取代码，主要少了数据的增强以及数据增强拼接，主要是想采用数据增强是否有用，结果是有一定的效果的
                label_data = Image.open(label_dir+name)
                la_array = np.array(label_data)
                name=name.replace('..png','..dcm')
                image = pydicom.read_file(image_dir+name)               
                im_array = image.pixel_array
                im_array = normalization(im_array)
                data.append(im_array)
                label.append(la_array)             
                 
    
        return np.array(data), np.array(label)
'''   
'''
test_readdata函数和上面的readdata类似，主要目的是方便区分测试和训练的，所以重新写了一个函数
   
    
def test_readdata(image_dir,label_dir,batch_size): ###测试读取数据
        data=[]
        label=[]
        pathDir = os.listdir(label_dir)   
        #print(pathDir)#取图片的原始路径
        filenumber=len(pathDir)
        #print(filenumber)
        sample = random.sample(pathDir, batch_size)  
        #print (sample)

        for name in sample:
                label_data = Image.open(label_dir+name)
                la_array = np.array(label_data)
                name=name.replace('..png','..dcm')
                image = pydicom.read_file(image_dir+name)    #######数据读取           
                im_array = image.pixel_array   #####数据矩阵读取
                im_array = normalization(im_array)
                data.append(im_array)  ####数据拼接
                label.append(la_array) ####标签拼接           
                 
    
        return np.array(data), np.array(label)

''' 


