import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    for i in range(1,201):
        path = '../original_pics/00'+str(i)+'.tif'
        print (path)
        img=cv2.imread(path)
        '''
        cv2.imshow('img', img)
        if cv2.waitKey(10000) & 0xff == 27:
            break
        '''
        #gauss = cv2.GaussianBlur(img,(3,3),0)
        mean = cv2.medianBlur(img, 15)
        edges = cv2.Canny(mean,100,200)
        lines = cv2.HoughLines(edges,1,np.pi/180,68)
        if type(lines) == type(None):
            print ('No lines found in the img')
        else:
            lines1 = lines[:,0,:]#提取为为二维
            for rho,theta in lines1[:]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.imshow('img', img)
        # press ESC to exit
        if cv2.waitKey(200) & 0xff == 27:
            break

if __name__ == '__main__':
    main();