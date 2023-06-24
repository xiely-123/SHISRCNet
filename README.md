# SHISRCNet: Super-resolution And Classification Network For Low-resolution Breast Cancer Histopathology Image
Luyuan Xie, Cong Li, Zirui Wang, Xin Zhang, Boyan Chen, Qingni Shen, and Zhonghai Wu
School of Software and Microelectronics, Peking University, Beijing

<p align="center">
   <img src="fig1.png" width="600"/>
</p>

## Organize the data (save data path and label to list):

### eg: in HR_groudtruth_test_classification_2.txt

                                            data path                                                                                              |    label

/home/xly/data/BreaKHis_v1/histology_slides_resize/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-16456/100X/SOB_M_MC-14-16456-100-051.png    |      1

## Training & Test
python train_SHISRCNet.py
