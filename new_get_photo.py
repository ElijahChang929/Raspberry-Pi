from picamera2 import Picamera2
import time
import numpy as np
import cv2
cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)
while(True):
    frame = cam.capture_array('main')
    cv2.imshow('Video Test',frame)
    if cv2.waitKey(1) == ord("q"):
        img = frame
        break
cam.stop()
cv2.destroyAllWindows()




