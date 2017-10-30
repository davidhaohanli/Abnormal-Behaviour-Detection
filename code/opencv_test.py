import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
	img=cv2.imread('../original_pics/001.tif')
	gauss = cv2.GaussianBlur(img,(3,3),0)
	edges = cv2.Canny(gauss,100,200)
	lines = cv2.HoughLines(edges,1,np.pi/180,68)
	if not lines.shape[0]:
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
	plt.imshow(img,cmap = 'gray')
	plt.show()

if __name__ == '__main__':
	main();
