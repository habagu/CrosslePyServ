#!/usr/bin/env python

import math
import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract

path = "C:/Users/GNTJONAU/Documents/Berufsschule/24-25/LF12/crossle-python/CrosslePyServ/sample_points.jpg"

img_src = cv2.imread(path)
# Resize to a specific width and height
new_width, new_height = 600, 900
img = cv2.resize(img_src, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

#correct image oreientation
blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

# Define the HSV range for green
lower_green = np.array([35, 100, 100])  # Lower bound of green
upper_green = np.array([85, 255, 255])  # Upper bound of green
mask = cv2.inRange(hsv,lower_green,upper_green)
res = cv2.bitwise_and(img,img,mask=mask)
_,thresh = cv2.threshold(res,125,255,cv2.THRESH_BINARY)

# check which are the best canny threshold values for your image
imgCanny = cv2.Canny(thresh, 180, 180)
dilate = cv2.dilate(imgCanny, None, iterations = 1)
# cv.imshow("dilate", dilate)
# cv.waitKey()

contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
        cv2.drawContours(img,cnt,-1,(60,255,255),4)

# loop over the contours add possible center points to point array
pts = []
for cnt in contours:
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    pts.append([cY, cX])



def distance_top_left (p):
    return math.sqrt((p[0] - 0)**2 + (p[1] - 0)**2)
def distance_top_right (p): 
    return math.sqrt((p[0] - 0)**2 + (p[1] - img.shape[1])**2)
def distance_bottom_left (p): 
    return math.sqrt((p[0] - img.shape[0])**2 + (p[1] - 0)**2)
def distance_bottom_right (p): 
    return math.sqrt((p[0] - img.shape[0])**2 + (p[1] - img.shape[1])**2)

        


p_top_left = min(pts, key = distance_top_left)
p_top_right = min(pts, key = distance_top_right)
p_bottom_left = min(pts, key = distance_bottom_left)
p_bottom_right = min(pts, key = distance_bottom_right)

# Draw points on the image
color = (255, 0, 255)  # Purple color in BGR  # Red color in BGR
radius = 10
thickness = -1  # Solid circle

'''
works fine until here
cv2.circle(img, (p_top_left[1], p_top_left[0]), radius, color, thickness)
cv2.circle(img, (p_top_right[1], p_top_right[0]), radius, color, thickness)
cv2.circle(img, (p_bottom_left[1], p_bottom_left[0]), radius, color, thickness)
cv2.circle(img, (p_bottom_right[1], p_bottom_right[0]), radius, color, thickness)

# Display the image
cv2.imshow("Image with Points", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# calculating the distance between points ( Pythagorean theorem ) 

height_1 = p_bottom_left[0] - p_top_left[0]
height_2 =p_bottom_right[0] - p_top_right[0]

width_1 = p_top_right[1] - p_top_left[1]
width_2 = p_bottom_right[1]- p_bottom_left[1]

max_height=max(int(height_1), int(height_2))
max_width = max(int(width_1), int(width_2))


# four input point 
input_pts=np.float32([p_top_left,p_top_right,p_bottom_left,p_bottom_right])

# output points for new transformed image
output_pts = np.float32([[0, 0],
                        [0, max_width],
                        [max_height , 0],
                        [max_height , max_width]])


# Compute the perspective transform M
h, status = cv2.findHomography(input_pts, output_pts)
img_oriented = cv2.warpPerspective(img, h, (max_width,max_height))

cv2.imshow("img_oriented", img_oriented)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray_oriented = cv2.cvtColor(img_oriented, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray_oriented, (11, 11), 0)

# Apply edge detection (Canny) to find edges in the img
edges = cv2.Canny(blurred, 0, 50)

#cv2.imshow("Detected Edges", edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ret, thresh = cv2.threshold(img_gray_oriented,127,255,cv2.THRESH_BINARY_INV)
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