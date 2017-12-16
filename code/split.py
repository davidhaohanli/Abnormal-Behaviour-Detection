import numpy as np

class Spliter(object):

    #TODO SET NORM
    normal = 120
    shapeParam = 3


    #TODO Default tuning
    def __init__(self,discardFloor=0.08,splitCeil=1.8,splitRemainderFloor=0.8,totalFullFillGate=0.6,shapeParamCeiling=1.5,\
                 shapeParamFloor=0.7):
        self.floor=discardFloor*Spliter.normal;
        self.ceil=splitCeil*Spliter.normal;
        self.remainderFloor=splitRemainderFloor*Spliter.normal
        self.totalGate=totalFullFillGate
        self.shapeCeil=shapeParamCeiling*Spliter.shapeParam
        self.shapeFloor=shapeParamFloor*Spliter.shapeParam

    def split(self,pos,fg_img,weight):
        posArea = self.areaCompute(pos,weight)
        realPos = np.zeros((0,5))
        for ind,area in enumerate(posArea):
            if area < self.floor:
                print('discard')
                continue
            if area > self.ceil:
                n = int(area // Spliter.normal)
                if area % Spliter.normal > self.remainderFloor:
                    n=n+1
                print('split',n)
                recArea = (pos[ind][0]-pos[ind][1])*(pos[ind][2]-pos[ind][3])
                shape = (pos[ind][0]-pos[ind][1])/(pos[ind][2]-pos[ind][3])
                step_y = (pos[ind][0] - pos[ind][1]) / n;
                step_x = (pos[ind][2] - pos[ind][3]) / n;
                if pos[ind][-1]/recArea < self.totalGate:
                    res=[]
                    #diganol splitting
                    for i in range(n):
                        pos1 = int(pos[ind][1] + i * step_y);
                        pos0 = int(pos[ind][1] + (i + 1) * step_y);
                        for j in range(n):
                            pos3 = int(pos[ind][3] + j * step_x);
                            pos2 = int(pos[ind][3] + (j + 1) * step_x);
                            res.append([pos0,pos1,pos2,pos3,fg_img[pos1:pos0,pos3:pos2].sum(),fg_img[pos1:pos0,pos3:pos2].mean()])
                    res.sort(key=lambda x:x[-1])
                    res=res[::-1]
                    for i in range(n):
                        print('diag chosen')
                        new = np.array(res[i][:-1])
                        realPos = np.concatenate((realPos, new.reshape(1, 5)), axis=0)
                elif shape > self.shapeCeil:
                    #y splitting
                    pos3=int(pos[ind][3])
                    pos2=int(pos[ind][2])
                    for i in range(n):
                        pos1 = int(pos[ind][1] + i * step_y);
                        pos0 = int(pos[ind][1] + (i + 1) * step_y);
                        print('y fullfilling')
                        new = np.array([pos0, pos1, pos2, pos3, fg_img[pos0:pos1, pos3:pos2].sum()])
                        realPos = np.concatenate((realPos, new.reshape(1,5)), axis=0)
                elif shape<self.shapeFloor:
                    #x splitting
                    pos1 = int(pos[ind][1])
                    pos0 = int(pos[ind][0])
                    for j in range(n):
                        pos3 = int(pos[ind][3] + j * step_x);
                        pos2 = int(pos[ind][3] + (j + 1) * step_x);
                        print('x fullfilling')
                        new = np.array([pos0, pos1, pos2, pos3, fg_img[pos0:pos1, pos3:pos2].sum()])
                        realPos = np.concatenate((realPos, new.reshape(1,5)), axis=0)
                else:
                    print ('reasonable shape, but fulfillment is high')
            else:
                realPos=np.concatenate((realPos,pos[ind].reshape(1,5)),axis=0)
        return realPos

    def areaCompute(self,pos,weight):
        area = np.zeros((pos.shape[0],1))
        for ind,eachPos in enumerate(pos):
            area[ind] = eachPos[-1]*weight[int((eachPos[0]+eachPos[1])//2)]
        return area