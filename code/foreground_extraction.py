# coding:utf8
import cv2

#version 1

def detect_video(video):
    camera = cv2.VideoCapture(video)
    history = 50    # 训练帧数

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0

    while True:
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)   # 获取 foreground mask

        if frames < history:
            frames += 1
            continue

        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 100, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 4)), iterations=2)
        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)

        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()

if __name__ == '__main__':
    video = '../ref_data/video/video.mp4'
    detect_video(video)

'''version 2
cap = cv2.VideoCapture('../video/video.mp4')

#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''
