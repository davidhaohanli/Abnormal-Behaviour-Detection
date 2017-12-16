# coding:utf8
import cv2
from poscal import poscal
from poscalflow import poscalflow
import scipy.io
import numpy as np
from weight_matrix import *
from split import *
from getFeatureUV import *

font=cv2.FONT_HERSHEY_COMPLEX
data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
u_seq_abnormal = data['u_seq_abnormal']
data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
v_seq_abnormal = data['v_seq_abnormal']
m = u_seq_abnormal.shape[0] # 图像的第一维度
n = u_seq_abnormal.shape[1] # 图像的第二维度

weight = Weight_matrix().get_weight_matrix()
spliter = Spliter()

## [weigh,sarea,sarea1,sth,stl]=weical();%先验知识的获得
list_names1 = ['../ref_data/fg_pics/' + str(i+1) + '.bmp' for i in range(200)]
list_names2 = ['../ref_data/original_pics/' + str(i+1).zfill(3) + '.tif' for i in range(200)]
list_names4 = ['../ref_data/ab_fg_pics/' + str(i+1) + '.bmp' for i in range(200)]

img3 = np.zeros((m,n,2))
label = np.ones((1,0))
datal = np.zeros((0,2))
for i in range(108,110):
    img1 = cv2.imread(list_names1[i])
    img2 = cv2.imread(list_names2[i])
    img4 = cv2.imread(list_names4[i])
    if img4 is None:
        img4 = np.zeros((m,n,3),dtype= np.uint8)
    img3[:,:,0] = u_seq_abnormal[:,:,i] * np.sqrt(weight).reshape((m,1))
    img3[:,:,1] = v_seq_abnormal[:,:,i] * np.sqrt(weight).reshape((m,1))
    imp1 = poscal(img1)
    realPos = spliter.split(imp1,img1,weight)
    ##################################################################################################### plot
    img = img2
    for i, item in enumerate(realPos):
        cv2.rectangle(img, (int(item[3]), int(item[1])), (int(item[2]), int(item[0])), (0, 0, 255))
        #cv2.putText(img, str(i), (int(item[3]), int(item[1]) - 5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('img', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    ########################################################################################################
    imp2 = poscal(img4)
    print(imp2)
    f1 = realPos.shape[0]
    f2 = imp2.shape[0]
    #print(imp1.shape)
    #print(imp2.shape)
    if imp2.max() == 0:
        label = np.concatenate((label,np.ones((1,f1))),axis = 1)
        data,im_s = poscalflow(img1,img3)
    else:
        '''
        dis = np.zeros((f2,f1))
        for k in range(f2):
            for j in range(f1):
                a1 = (imp1[j,0]+imp1[j,1])//2 - (imp2[k,0]+imp2[k,1])//2
                a2 = (imp1[j,2]+imp1[j,3])//2 - (imp2[k,2]+imp2[k,3])//2
                a = np.array([a1,a2])
                dis[k,j] = np.sqrt(np.dot(a,a))
        k = np.argmin(dis)
                print(data)
        data = np.concatenate((data[:,0:k],data[:,(k+1):]),axis = 1)
        print(data)
        '''
        label = np.concatenate((label, np.ones((1, f1))), axis=1)
        data = getFeaturesUV(realPos,u_seq_abnormal[:,:,i],v_seq_abnormal[:,:,i])

    datal = np.concatenate((datal,data),axis=0)


    ########################################################
''' data,pos = poscalflow(img1,img3)
    _,abnormal_pos = poscalflow(img4,img3)
    data=data.T
    labels = np.zeros(data.shape[0]).reshape(-1,1)
    for checker in abnormal_pos:
        labels = np.all(pos==checker,axis=1).reshape(-1,1)


    ########################################################
'''