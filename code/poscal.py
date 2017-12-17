# coding:utf8
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from split import *
from weight_matrix import *


font=cv2.FONT_HERSHEY_COMPLEX

def poscal(img):
    img = img[:,:,0]

    m,n = img.shape
    kernel = np.ones((6,1),np.uint8)
    im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)    # 开运算
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # 闭运算
    im_labels = measure.label(im,neighbors=8,connectivity=1)       #从0开始连通域标记-8

    num = im_labels.max()     # 标记连通域的个数（除去背景连通域）

    if num==0:
        im_s = np.zeros((1,5))   # 考虑一张全黑图的情况 如果不考虑 可以略过
    else:
        im_s = np.zeros((num,5))
        for i in range(num):
            temp = np.copy(im_labels)
            temp[temp != (i+1)]=0
            index = np.where(temp ==(i+1))
            im_s[i,0]= max(index[0]) #person's foot y_val
            im_s[i,1]= min(index[0]) #person's head y_val
            im_s[i,2]= max(index[1]) #person's right side x_val
            im_s[i,3]= min(index[1]) #person's left side x_val
            im_s[i,4]= len(index[0]) #area of the person
    return im_s,im

def main():
    img = cv2.imread('../ref_data/fg_pics/1.bmp')
    im_s,im = poscal(img)
    weight = Weight_matrix().get_weight_matrix()
    im_s = Spliter().split(im_s,im,weight)
    #np.savetxt('../ref_data/connectedFieldImg.txt',im_s,delimiter=',')
    print(im_s)
    #plot
    originalimg = cv2.imread('../ref_data/original_pics/001.tif')
    for i,item in enumerate(im_s):
        cv2.rectangle(originalimg,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0, 0, 255))
        cv2.putText(originalimg, str(i), (int(item[3]),int(item[1])-5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('originalimg',originalimg)
    cv2.imshow('img', img)
    cv2.imshow('afterProcessing: ',im)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows();

if __name__=='__main__':
    main()

'''
a=bwconncomp(im);
im_flag=bwlabel(im,a.Connectivity);%标志二值图中连通域
if a.NumObjects==0
    im_s=[];
else
    for i=1:a.NumObjects
        im_f=im_flag;
        im_f(~(im_f==i))=0;
        [p,q]=find(im_f);
        im_s(i,1)=max(p);
        im_s(i,2)=min(p);
        im_s(i,3)=max(q);
        im_s(i,4)=min(q);
        im_s(i,5)=length(p);
    end
end
'''