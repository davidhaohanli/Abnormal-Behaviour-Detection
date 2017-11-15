import numpy as np
import cv2
import lxml.etree as et
import re

from xmlLoader_generator import *

def proportion(pic):
    res=[]
    if pic is not None:
        for y in pic:
            res.append((int(y.get('val')),abs(int(re.findall('\(.*,',y[0].text)[0][1:-1])-\
                       int(re.findall('\(.*,',y[-1].text)[0][1:-1]))))
    return res;

def weight(tps,y):
    y1=tps[0][0]
    y2=tps[-1][0]
    ab=tps[0][1];
    cd=tps[-1][1];
    return (y-y1)/(y2-y1)+(y2-y)/(y2-y1)*(ab/cd)

def main ():
    pic = Poi_handle().searchPic(input('no. of hough lines pic intend to use as weight generator: \n'))
    while 1:
        y=input('y val: \n')
        print(weight(proportion(pic),int(y)))
if __name__ == '__main__':
    main()