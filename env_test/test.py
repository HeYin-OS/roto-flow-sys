import cv2
import numpy as np
import cv2
print(cv2.__version__)

img = np.zeros((512, 512, 3), np.uint8)
while True:
    num = cv2.waitKey(0)
    cv2.imshow('img', img)
    print(num)