# coding:utf8
import cv2
import numpy as np
from skimage import measure,color
import matplotlib.pyplot as plt

img = cv2.imread('../fg_pics/1.bmp')
img = img[:,:,0]

m,n = img.shape
kernel = np.ones((6,1),np.uint8)
im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)    # 开运算
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # 闭运算
im_labels = measure.label(im,connectivity=2)       #从0开始连通域标记-8

num = im_labels.max()+1
im_s = np.zeros((num,5))

temp = im_labels
temp[temp != 17] = 0

index = np.where(temp==17)
print(index[0])
print(max(index[0]))

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