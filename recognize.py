import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

def num(x):
    return x



cv.namedWindow('Control Panel')
cv.createTrackbar('H:min','Control Panel',0,255,num)
cv.createTrackbar('H:max','Control Panel',0,255,num)
cv.createTrackbar('S:min','Control Panel',0,255,num)
cv.createTrackbar('S:max','Control Panel',0,255,num)
cv.createTrackbar('V:min','Control Panel',0,255,num)
cv.createTrackbar('V:max','Control Panel',0,255,num)
cv.createTrackbar('C1','Control Panel',0,255,num)
cv.createTrackbar('C2','Control Panel',0,255,num)
cv.createTrackbar('C3','Control Panel',0,255,num)


kernel = np.ones((3,3),np.uint8)  
while True:
    img = cv.imread('sudoku4.tif')
    l_h = cv.getTrackbarPos('H:min','Control Panel')
    h_h = cv.getTrackbarPos('H:max','Control Panel')
    l_s = cv.getTrackbarPos('S:min','Control Panel')
    h_s = cv.getTrackbarPos('S:max','Control Panel') 
    l_v = cv.getTrackbarPos('V:min','Control Panel')
    h_v = cv.getTrackbarPos('V:max','Control Panel')
    c1 = cv.getTrackbarPos('C1','Control Panel') 
    c2 = cv.getTrackbarPos('C2','Control Panel')
    c3 = cv.getTrackbarPos('C3','Control Panel')
    hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
    image_mask_p = cv.inRange(hsv,np.array([l_h,l_s,0]),np.array([h_h,h_s,255]))
    HSV_p = cv.bitwise_and(img,img,mask = image_mask_p)

    cv.imshow('close',HSV_p)

    if cv.waitKey(1) == ord("q"):
        break
cv.destroyAllWindows()
