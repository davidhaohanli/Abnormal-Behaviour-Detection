from plot_hough_lines import *
import cv2
import numpy as np


font=cv2.FONT_HERSHEY_COMPLEX

def nothing(x):
    pass;

def crossPoints(img,y,):
    temp = np.copy(img)
    pt = [];
    for x in range(img.shape[1]):
        if img[y][x].any() > 0:
            pt.append(x)
    if len(pt) > 0:
        cv2.putText(temp, str(pt[0]), (20, 50), font, 1, (255, 255, 0), 1)
    if len(pt) > 1:
        cv2.putText(temp, str(pt[-1]), (20, 100), font, 1, (255, 255, 0), 1)
    # print(len(pt))
    return temp,pt

def main():
    hough_main(True)
    img = cv2.imread('../ref_data/hough_lines_only/lines_only_' + input('No. of pic:\n').zfill(3) + '.tif')
    cv2.namedWindow('img')
    cv2.createTrackbar('y_val','img',0,img.shape[0]-1,nothing)
    y_pre='dummy'
    while 1:
        #print ('in')
        x1=0
        y1=cv2.getTrackbarPos('y_val', 'img')
        x2=img.shape[1]
        y2=y1
        if y1 != y_pre:
            temp,pt=crossPoints(img,y1)
            cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 255), 1)
        y_pre=y1
        cv2.imshow('img',temp)
        if cv2.waitKey(1) & 0xFF == 27:
            if len(pt) >= 2:
                np.savetxt('../ref_data/poi',np.array([y1,pt[0],pt[-1]]),delimiter=',')
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()