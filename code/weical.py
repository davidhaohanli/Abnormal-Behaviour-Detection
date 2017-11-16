import numpy as np
import cv2


def main():
    for i in range(1,200):
        im = cv2.imread('../img_optical_flow/'+str(i)+'.bmp')
        cv2.imshow('img',im)
        if cv2.waitKey(200) & 0xff == 27:
            break
        #cv2.destroyAllWindows()


if __name__=='__main__':
    main()