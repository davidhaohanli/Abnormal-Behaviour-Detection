import cv2
import numpy as np
#import matplotlib.pyplot as plt

def hough_lines(img):
    return cv2.HoughLines(cv2.Canny(img,105,130),1,np.pi/180,89)

def plot_hough_lines(lines,img):
    if lines is not None:
        lines1 = lines[:, 0, :]
        for rho, theta in lines1[:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    else:
        print ('Hough line not find')
    return img;

def main():
    for i in range(1,201):
        img = cv2.imread('../original_pics/'+str(i).zfill(3)+'.tif')
        lines = hough_lines(img)
        new_img = plot_hough_lines(lines,img)
        #new_img = plot_hough_lines(lines,np.zeros((img.shape[0],img.shape[1],3)))
        cv2.imshow('img', new_img)
        # press ESC to exit
        cv2.imwrite('../ref_data/pics_with_hough_lines/'+str(i).zfill(3)+'.tif',new_img)
        if cv2.waitKey(200) & 0xff == 27:
            break

if __name__ == '__main__':
    main();