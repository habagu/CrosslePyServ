#!/usr/bin/env python

import math
import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract

path = "C:/Users/GNTJONAU/Documents/Berufsschule/24-25/LF12/crossle-python/CrosslePyServ/sample.jpg"

img = cv2.imread(path)
img_gray = cv2.imread(path,0)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray, (11, 11), 0)

# Apply edge detection (Canny) to find edges in the img
edges = cv2.Canny(blurred, 0, 50)

#cv2.imshow("Detected Edges", edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ret, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
thresh2 = cv2.bitwise_not(thresh)

contours,hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, 1)

max_area = -1

# find contours with maximum area
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt,True), True)
    if cv2.contourArea(cnt) > max_area:
        max_area = cv2.contourArea(cnt)
        max_cnt = cnt
        max_approx = approx

# cut the crossword region, and resize it to a standard size of 130x130
x,y,w,h = cv2.boundingRect(max_cnt)
cross_rect = thresh2[y:y+h, x:x+w]

cv2.imshow("largest contour", cross_rect)

cv2.waitKey()