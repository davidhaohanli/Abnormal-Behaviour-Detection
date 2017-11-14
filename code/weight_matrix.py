from plot_hough_lines import *

def main():
    hough_main(True)
    img = cv2.imread('../ref_data/hough_lines_only/lines_only_' + input('No. of pic:\n').zfill(3) + '.tif')
    x1=0
    y1=img.shape[0]//2;
    x2=img.shape[1]
    y2=y1
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

    cv2.imshow('img', img)
    cv2.waitKey(20000)

    pass;

if __name__ == '__main__':
    main()