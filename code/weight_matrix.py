import re
import numpy as np
from xmlLoader_generator import *

def diff(pic):
    res = []
    if pic is not None:
        for y in pic:
            res.append((int(y.get('val')), abs(int(re.findall('\(.*,', y[0].text)[0][1:-1]) - \
                                               int(re.findall('\(.*,', y[-1].text)[0][1:-1]))))
    return res;

class Weight_matrix:

    def __init__(self,tps=diff(Poi_handle().searchPic(3))):
        self.y1 = tps[0][0]
        self.y2 = tps[-1][0]
        self.ab = tps[0][1];
        self.cd = tps[-1][1];
        #print (self.y1)
        self.compute_weight_matrix()

    def y_weight(self,y):
        return (y - self.y2) / (self.y1 - self.y2) + (self.y1 - y) / (self.y1 - self.y2) * (self.ab / self.cd)

    def compute_weight_matrix(self):
        self.weight_matrix=np.vectorize(self.y_weight)(np.arange(158))

    def get_weight_matrix(self):
        return self.weight_matrix

def test ():
    n=input('no. of hough lines pic intend to use as weight generator, \'d\' for default: \n')
    if n == 'd':
        print (Weight_matrix().get_weight_matrix())
    else:
        print(Weight_matrix(diff(Poi_handle().searchPic(n))).get_weight_matrix())

if __name__ == '__main__':
    test()