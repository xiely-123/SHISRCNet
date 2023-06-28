# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:11:41 2020

@author: ALW
"""

import torch   #########调用pytorch 
import torch.nn as nn   ###############调用pytorch中的nn函数定义为nn pyortch自带，引入CrossEntropyLoss（CE准则）
#import model as ml    
from torch import optim   ######调用pytorch中的optim，主要目的是引入优化器
import dataread       ######调用dcm_dataread文件，主要目的是dcm数据的读取
from torch.autograd import Variable   ###############调用pytorch中的Variable，主要目的是数据和标签的接口
import time  #####引入时间函数time
import os    #####引入显卡读取函数os
import numpy as np
#import loss
#import untils
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import SRResNet_class as SRmodel_res
import dataloader_torch_SRC as dl
import torch.utils.data as Data
import loss_FL

os.environ['CUDA_VISIBLE_DEVICES'] = '1'   ####gpu选择，可以修改设置GPU数量，2,3等数字是GPU序号
batch_size = 4
batch_size1 = 10
lr = 1e-3

net = SRmodel_res.SIHSRCNet()
#print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ###如果没有gpu选择cpu
#net = torch.load("/home/xly/SR_classification/SHISRCNet_v2/SHISRCNet/18my_model.pth").module  
if torch.cuda.device_count() > 1:
  net = nn.DataParallel(net)  ####gpu并行训练
# Assuming that we are on a CUDA machine, this should print a CUDA device:
net.to(device)  ####将网络放入gpu中进行训练

print(device)###显示gpu是否占用上，cuda:0--表示GPU已正常使用



LABELS = torch.cat([torch.arange(batch_size) for i in range(1)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() #one-hot representations
LABELS = LABELS.to(device)

def ntxent_loss(features, features_1, temp=2):
    """
    NT-Xent Loss.

    Args:
    z1: The learned representations from first branch of projection head
    z2: The learned representations from second branch of projection head
    Returns:
    Loss
    """
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(device)
    labels = LABELS[~mask].view(LABELS.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temp
    return logits, labels




#net=ml.Unet(1,2).to(device)
net=net.double()  #更改net数据类型double
criterion = torch.nn.L1Loss()#torch.nn.MSELoss()#torch.nn.SmoothL1Loss()####nn.CrossEntropyLoss()  #####loss函数采用CE准则loss，Torch自带的CrossEntropyLoss
criterion2 = loss_FL.FocalLoss(2)#nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)  ########优化器采用Adam，学习率采用1e-4，Torch自带的Adam


###SR
Train_list = np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_train_classification_2.txt", dtype=str, delimiter='  ')[1:]
Test_list = np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_test_classification_2.txt", dtype=str, delimiter='  ')#[:20]
Train_list = Train_list#[:10]
Test_list = Test_list#[:10]
print(Train_list.shape,Test_list.shape)
torch_dataset = dl.SPDataSet(Train_list)
loader = Data.DataLoader(dataset = torch_dataset, batch_size = batch_size, shuffle = True,num_workers = 4,pin_memory=True,) 
torch_dataset_test = dl.SPDataSet(Test_list)
loader_test = Data.DataLoader(dataset = torch_dataset_test, batch_size = 1, shuffle = True,num_workers = 4,pin_memory=True,) 

##class
Test_list_40X = np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_test_classification_40X.txt", dtype=str, delimiter='  ')#[:10]
torch_dataset_test_40X = dl.SPDataSet(Test_list_40X)
loader_test_40X = Data.DataLoader(dataset = torch_dataset_test_40X, batch_size = 10, shuffle = True,num_workers = 8,pin_memory=True,) 

Test_list_100X = np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_test_classification_100X.txt", dtype=str, delimiter='  ')#[:10]
torch_dataset_test_100X = dl.SPDataSet(Test_list_100X)
loader_test_100X = Data.DataLoader(dataset = torch_dataset_test_100X, batch_size = 10, shuffle = True,num_workers = 8,pin_memory=True,) 

Test_list_200X = np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_test_classification_200X.txt", dtype=str, delimiter='  ')#[:10]
torch_dataset_test_200X = dl.SPDataSet(Test_list_200X)
loader_test_200X = Data.DataLoader(dataset = torch_dataset_test_200X, batch_size = 10, shuffle = True,num_workers = 8,pin_memory=True,) 


Test_list_400X= np.loadtxt("/home/xly/SR_classification/classification/HR_groudtruth_test_classification_400X.txt", dtype=str, delimiter='  ')#[:10]
torch_dataset_test_400X = dl.SPDataSet(Test_list_400X)
loader_test_400X = Data.DataLoader(dataset = torch_dataset_test_400X, batch_size = 10, shuffle = True,num_workers = 8,pin_memory=True,) 


all_time = 0          #####一共所用时间
num_epochs = 1000  #######训练轮数

#print(net)
#net.load_state_dict(checkpoint)

start_epoch = 0
for epoch in range(start_epoch,num_epochs):
    start_time = time.time() 
    if epoch%2 == 0:   ####学习率衰减
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            lr = lr * 0.95
    #print(epoch ,param_group['lr'])
   
    net.train()
    loss_ave = torch.zeros(1).squeeze().cuda()
    loss_ave_re = torch.zeros(1).squeeze().cuda()
    for i,dataset in enumerate(loader):
        
        batchLR,batchHR,batchLabel = dataset
        
        inputs1 = Variable(batchLR).to(device)  #######Train
        target_SR = Variable(batchHR).to(device)
        target_class = Variable(batchLabel).to(device)
        outSR, outClass, feature ,outClass_1, feature_1= net(inputs1,target_SR)
        #print(out1.shape)
        #out = out.reshape(batch_size,-1)
        #target = target.reshape(batch_size,-1)
        logits, labels = ntxent_loss(feature,feature_1)
        loss = 0.6*criterion(outSR,target_SR) + 0.3*(0.5*criterion2(outClass,target_class)+0.5*criterion2(outClass_1,target_class)) + 0.1*criterion1(logits, labels) #####out*inputs1.double()
        #0.8*criterion(outSR,target_SR) + 0.2*criterion2(outClass,target_class)
        optimizer.zero_grad()           #归零梯度，每次反向传播前都要归零梯度，不然梯度会累积，造成结果不收敛
        loss.backward()                 #反向传播
        optimizer.step()                #更新参数
        train_time = time.time() - start_time 
        loss_ave = loss_ave + loss.item()
        #loss_ave_re = loss_ave_re + loss_re.item()
        print('\r','Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ave_loss:{:.6f},ave_loss_re:{:.6f},time:{:.3f},leanring_rate:{:.6f}'.format(epoch + 1, num_epochs, i + 1, int(len(Train_list)/batch_size)+1, loss.item(), loss_ave/(i+1), loss_ave/(i+1), train_time, param_group['lr']),end='')
    print("         ")
    torch.save(net, "./SHISRCNet_x4/"+str(epoch + 1)+"my_model.pth")###model_SRResNet_x8_L1 -> 4
    
    """
    Test
    """
    
    net.eval()
    with torch.no_grad():
        all_PSNR = 0
        all_SSIM = 0
        for i,dataset in enumerate(loader_test):
            
            batchLR,batchHR,_ = dataset
            inputs1 = Variable(batchLR).to(device)  #######Train
            target_SR = Variable(batchHR).to(device)
            out= net(inputs1,target_SR)[0].cpu().numpy().squeeze(0)
            batchHR = batchHR.numpy().squeeze(0)
            #print(batchHR.shape,out.shape)
            single_PSNR = compare_psnr(batchHR.T ,out.T )#.astype('uint8')
            all_PSNR = single_PSNR + all_PSNR
            all_SSIM = compare_ssim(batchHR.T ,out.T , multichannel=True) + all_SSIM
        print('\r','ave_PSNR:{:.6f}  all_SSIM:{:.6f}'.format(all_PSNR/(i+1), all_SSIM/(i+1)),end='') 
        print("         ")
        
        
        
        
        running_corrects_test = torch.zeros(1).squeeze().cuda()      
        for i,dataset in enumerate(loader_test_40X):
            
            image,batchHR,label = dataset 
            inputs1 = Variable(image).to(device)  #######Train
            target_SR = Variable(batchHR).to(device)
            target = Variable(label).to(device)
            _,out,_,_,_= net(inputs1,target_SR)
            _ , prediction =  torch.max(out,1)
            running_corrects_test = running_corrects_test + torch.sum(prediction == target)
            #print(torch.sum(prediction == target))
        print('\r','X40ACC:{:.6f}'.format(running_corrects_test/(len(Test_list_40X))),end='')  
        print("         ")    
        
        running_corrects_test = torch.zeros(1).squeeze().cuda()      
        for i,dataset in enumerate(loader_test_100X):
            
            image,batchHR,label = dataset 
            inputs1 = Variable(image).to(device)  #######Train
            target_SR = Variable(batchHR).to(device)
            target = Variable(label).to(device)
            _,out,_,_,_= net(inputs1,target_SR)
            _ , prediction =  torch.max(out,1)
            running_corrects_test = running_corrects_test + torch.sum(prediction == target)
            #print(torch.sum(prediction == target))
        print('\r','X100ACC:{:.6f}'.format(running_corrects_test/(len(Test_list_100X))),end='')  
        print("         ")          

        running_corrects_test = torch.zeros(1).squeeze().cuda()      
        for i,dataset in enumerate(loader_test_200X):
            
            image,batchHR,label = dataset 
            inputs1 = Variable(image).to(device)  #######Train
            target_SR = Variable(batchHR).to(device)
            target = Variable(label).to(device)
            _,out,_,_,_= net(inputs1,target_SR)
            _ , prediction =  torch.max(out,1)
            running_corrects_test = running_corrects_test + torch.sum(prediction == target)
            #print(torch.sum(prediction == target))
        print('\r','X200ACC:{:.6f}'.format(running_corrects_test/(len(Test_list_200X))),end='')  
        print("         ")          
        running_corrects_test = torch.zeros(1).squeeze().cuda()
        for i,dataset in enumerate(loader_test_400X):
            
            image,batchHR,label = dataset 
            inputs1 = Variable(image).to(device)  #######Train
            target_SR = Variable(batchHR).to(device)
            target = Variable(label).to(device)
            _,out,_,_,_= net(inputs1,target_SR)
            _ , prediction =  torch.max(out,1)
            running_corrects_test = running_corrects_test + torch.sum(prediction == target)
            #print(torch.sum(prediction == target))
        print('\r','X400ACC:{:.6f}'.format(running_corrects_test/(len(Test_list_400X))),end='')  
        print("         ")  
        