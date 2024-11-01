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

 # Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)


linesP = cv2.HoughLinesP(edges, 1, np.pi / 1800, 100, None, 1, 1)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

#cv2.imshow("Source", img)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv2.waitKey()