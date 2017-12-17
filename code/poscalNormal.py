# coding:utf8
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

font=cv2.FONT_HERSHEY_COMPLEX

def poscalNormal(img1,img4):
    m,n,_ = img1.shape
    if img4 is None:
        img4 = np.zeros((m, n, 3), dtype=np.uint8)
    img1 = img1[:,:,0]
    img4 = img4[:,:,0]

    m,n = img1.shape
    kernel = np.ones((6,1),np.uint8)
    #############################################################################
    img1[img4 != 0] = 0
    #############################################################################
    im = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)    # 开运算
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # 闭运算
    im_labels = measure.label(im,connectivity=1,neighbors= 8)       #从0开始连通域标记-8


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
    img1 = cv2.imread('../ref_data/fg_pics/169.bmp')
    img4 = cv2.imread('../ref_data/ab_fg_pics/169.bmp')

    im_s,_ = poscalNormal(img1,img4)
    print(im_s)
    #plot
    img = cv2.imread('../ref_data/original_pics/169.tif')
    for i,item in enumerate(im_s):
        cv2.rectangle(img,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0, 0, 255))
        cv2.putText(img, str(i), (int(item[3]),int(item[1])-5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('img',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()