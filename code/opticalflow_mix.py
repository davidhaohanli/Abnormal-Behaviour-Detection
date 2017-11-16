import cv2
import numpy as np

list_names = ['../original_pics/' + str(i+1).zfill(3) + '.tif' for i in range(200)]

frame1 = cv2.imread(list_names[0])
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
counter = 1

while counter < len(list_names):
    frame2 = cv2.imread(list_names[counter])
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)

    prvs = next.copy()

    if counter == len(list_names)-1:
        k = cv2.waitKey(0) & 0xff
    else: # Else, wait for 1 second for a key
        k = cv2.waitKey(1000) & 0xff

    if k == 27:
        break
    elif k == ord('s'): # Change
        cv2.imwrite('opticalflow_horz' + str(counter) + '-' + str(counter+1) + '.pgm', horz)
        cv2.imwrite('opticalflow_vert' + str(counter) + '-' + str(counter+1) + '.pgm', vert)

    # Increment counter to go to next frame
    counter +=1


cv2.destroyAllWindows()