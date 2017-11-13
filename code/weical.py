import numpy as np
import cv2


def main():
    for i in range(1,201):
        im = cv2.imread('../original_pics/'+str(i).zfill(3)+'.tif')

        cv2.imshow('img',im)
        if cv2.waitKey(200) & 0xff == 27:
            break
        #cv2.destroyAllWindows()




if __name__=='__main__':
    main()