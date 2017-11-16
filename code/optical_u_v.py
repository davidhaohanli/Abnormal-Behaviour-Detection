import cv2
import numpy as np

# Create list of names here from my1.bmp up to my20.bmp
list_names = ['../original_pics/' + str(i+1).zfill(3) + '.tif' for i in range(200)]

# Read in the first frame
frame1 = cv2.imread(list_names[0])
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# Set counter to read the second frame at the start
counter = 1

# Until we reach the end of the list...
while counter < len(list_names):
    # Read the next frame in
    frame2 = cv2.imread(list_names[counter])
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(prvs, next,None,0.5,3,15,3,5,1.2,0)

    # Normalize horizontal and vertical components
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')

    # Show the components as images
    cv2.imshow('Horizontal Component', horz)
    cv2.imshow('Vertical Component', vert)

    # Change - Make next frame previous frame
    prvs = next.copy()

    # If we get to the end of the list, simply wait indefinitely
    # for the user to push something
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
    counter += 1

cv2.destroyAllWindows()