# coding:utf8
import cv2
import numpy as np
from skimage import measure,color
import matplotlib.pyplot as plt
import scipy.io

def poscalflow(img1,img3):
    m,n,l= img3.shape
    img = img1[:,:,0]

    kernel = np.ones((6,1),np.uint8)
    im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)    # 开运算
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # 闭运算
    im_labels = measure.label(im,connectivity=2)       #从0开始连通域标记-8

    num = im_labels.max()     # 标记连通域的个数（除去背景连通域）

    if num==0:
        data = np.zeros(2)
        im_s = np.zeros((1,5))   # 考虑一张全黑图的情况 如果不考虑 可以略过
    else:
        im_s = np.zeros((num,5))
        data = np.zeros((2,num))
        for i in range(num):
            temp = np.copy(im_labels)
            temp[temp != (i+1)]=0
            data1 = np.zeros(l)
            index = np.where(temp == (i + 1))
            for j in range(l):
                iml = img3[:,:,j]
                aa = iml[temp==(i+1)]
                aaa = index[0]          # 第一坐标的索引值
                aa_f = np.sort(np.unique(aaa))
                aa[aaa <= aa_f[1]] = -1
                aa[aaa>= aa_f[len(aa_f)-3]] = -1
                aa= aa[np.where(aa != (-1))]
                data1[j]= np.mean(aa)
            data[:,i] = data1
            im_s[i, 0] = max(index[0])
            im_s[i, 1] = min(index[0])
            im_s[i, 2] = max(index[1])
            im_s[i, 3] = min(index[1])
            im_s[i, 4] = len(index[0])
    return data,im_s

def main():
    data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
    u_seq_abnormal = data['u_seq_abnormal']
    data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
    v_seq_abnormal = data['v_seq_abnormal']

    m, n, number = u_seq_abnormal.shape
    img3 = np.zeros((m, n, 2))
    img3[:, :, 0] = u_seq_abnormal[:, :, 0]
    img3[:, :, 1] = v_seq_abnormal[:, :, 0]
    img1 = cv2.imread('../ref_data/fg_pics/1.bmp')
    data, im_s = poscalflow(img1,img3)
    print(data)
    print(im_s)

if __name__ == '__main__':
    main()

    '''
        index = np.where(temp ==(i+1))
        im_s[i,0]= max(index[0])
        im_s[i,1]= min(index[0])
        im_s[i,2]= max(index[1])
        im_s[i,3]= min(index[1])
        im_s[i,4]= len(index[0])'''