import numpy as np
import cv2
import lxml.etree as et
import re

from xmlLoader_generator import *

def diff(pic):
    res=[]
    if pic is not None:
        for y in pic:
            res.append((int(y.get('val')),abs(int(re.findall('\(.*,',y[0].text)[0][1:-1])-\
                       int(re.findall('\(.*,',y[-1].text)[0][1:-1]))))
    return res;


def y_weight(y,tps=diff(Poi_handle().searchPic(3))):
    y1=tps[0][0]
    y2=tps[-1][0]
    ab=tps[0][1];
    cd=tps[-1][1];
    return (y-y2)/(y1-y2)+(y1-y)/(y1-y2)*(ab/cd)

def test ():
    n=input('no. of hough lines pic intend to use as weight generator, \'d\' for default: \n')
    while 1:
        y=input('y val: \n')
        if n == 'd':
            print (y_weight(int(y)))
        else:
            print(y_weight(int(y),diff(Poi_handle().searchPic(n))))
if __name__ == '__main__':
    test()