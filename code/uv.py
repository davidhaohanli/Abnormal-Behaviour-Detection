# coding:utf8
import cv2
from poscal import poscal
from poscalflow import poscalflow
import scipy.io
import numpy as np
from weight_matrix import *
from split import *
from getFeatureUV import *
from poscalNormal import *
from Classifiers import *

font = cv2.FONT_HERSHEY_COMPLEX
data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
u_seq_abnormal = data['u_seq_abnormal']
data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
v_seq_abnormal = data['v_seq_abnormal']
m = u_seq_abnormal.shape[0]  # y-dim of pictures
n = u_seq_abnormal.shape[1]  # x-dim of pictures

weight = Weight_matrix().get_weight_matrix()
spliter = Spliter()

list_names1 = ['../ref_data/fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]
list_names2 = ['../ref_data/original_pics/' + str(i + 1).zfill(3) + '.tif' for i in range(200)]
list_names4 = ['../ref_data/ab_fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]

img3 = np.zeros((m, n, 2))
datal = np.zeros((0, 2))
datalAb = np.zeros((0, 2))

for i in range(199):
    print('img: ', i)
    img1 = cv2.imread(list_names1[i])
    img2 = cv2.imread(list_names2[i])
    img4 = cv2.imread(list_names4[i])
    if img4 is None:
        img4 = np.zeros((m, n, 3), dtype=np.uint8)
    img3[:, :, 0] = u_seq_abnormal[:, :, i] * np.sqrt(weight).reshape((m, 1))
    img3[:, :, 1] = v_seq_abnormal[:, :, i] * np.sqrt(weight).reshape((m, 1))
    imp1, im = poscalNormal(img1, img4)
    realPos = spliter.split(imp1, im, weight)
    #####################################################################################################

    # plot in original pictures
    for i, item in enumerate(realPos):
        cv2.rectangle(img2, (int(item[3]), int(item[1])), (int(item[2]), int(item[0])), (0, 0, 255))
        #cv2.putText(img, str(i), (int(item[3]), int(item[1]) - 5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('img', img2)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    ########################################################################################################
    imp2, _ = poscal(img4)
    f1 = realPos.shape[0]
    data = np.zeros((f1, 2))
    f2 = imp2.shape[0]

    if imp2.max() == 0:
        data, im_s = poscalflow(img1, img3)
        dataAb = np.zeros((0, 2))
    else:
        data = getFeaturesUV(realPos, u_seq_abnormal[:, :, i], v_seq_abnormal[:, :, i])
        dataAb = np.zeros((1, 2))
        dataAb = getFeaturesUV(imp2, u_seq_abnormal[:, :, i], v_seq_abnormal[:, :, i])
    datal = np.concatenate((datal, data), axis=0)
    datalAb = np.concatenate((datalAb, dataAb), axis=0)
train_data = np.concatenate((datal, datalAb), axis=0)
train_data = np.nan_to_num(train_data)

train_label = np.concatenate((np.zeros(datal.shape[0]), np.ones(datalAb.shape[0])), axis=0)

# classifiers = Classifiers(train_data,train_label)