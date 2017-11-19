# coding:utf8
import cv2
import numpy as np
from skimage import measure,color
import matplotlib.pyplot as plt
import scipy.io

data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
u_seq_abnormal = data['u_seq_abnormal']
data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
v_seq_abnormal = data['v_seq_abnormal']

m,n,number = u_seq_abnormal.shape
img3 = np.zeros((m,n,2))
img3[:,:,0] = u_seq_abnormal[:,:,0]
img3[:,:,1] = v_seq_abnormal[:,:,0]
m,n,l= img3.shape

img1 = cv2.imread('../fg_pics/1.bmp')
img = img1[:,:,0]

kernel = np.ones((6,1),np.uint8)
im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)    # 开运算
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # 闭运算
im_labels = measure.label(im,connectivity=2)       #从0开始连通域标记-8

num = im_labels.max()     # 标记连通域的个数（除去背景连通域）

if num==0:
    im_s = np.zeros(5)   # 考虑一张全黑图的情况 如果不考虑 可以略过
else:
    # im_s = np.zeros((num,5))
    for i in range(num):
        temp = np.copy(im_labels)
        temp[temp != (i+1)]=0
        for j in range(l):
            iml = img3[:,:,j]
            index = np.where(temp==(i+1))
            aa = iml[temp==(i+1)]
            aaa = index[0]          # 第一坐标的索引值
            print(aaa.shape)
            aa_f = np.sort(np.unique(aaa))
            aa[aaa <= aa_f[1]] = -1
            aa[aaa>= aa_f[len(aa_f)-3]] = -1









        '''
        index = np.where(temp ==(i+1))
        im_s[i,0]= max(index[0])
        im_s[i,1]= min(index[0])
        im_s[i,2]= max(index[1])
        im_s[i,3]= min(index[1])
        im_s[i,4]= len(index[0])'''