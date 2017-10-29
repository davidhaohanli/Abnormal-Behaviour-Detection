import cv2

a=cv2.imread('../original_pics/001.tif')

cv2.imshow('a',a)

if cv2.waitKey(20000) & 0xff == 27:
    pass;