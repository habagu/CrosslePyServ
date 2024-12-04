#!/usr/bin/env python

import math
import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract as tes

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

#works fine until here
cv2.circle(img, (p_top_left[1], p_top_left[0]), radius, color, thickness)
cv2.circle(img, (p_top_right[1], p_top_right[0]), radius, color, thickness)
cv2.circle(img, (p_bottom_left[1], p_bottom_left[0]), radius, color, thickness)
cv2.circle(img, (p_bottom_right[1], p_bottom_right[0]), radius, color, thickness)
'''
# Display the image
cv2.imshow("Image with Points", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
production
p_top_left = 
p_top_right = 
p_bottom_left = 
p_bottom_right = 
'''

# calculating the height/width

height_1 = p_bottom_left[0] - p_top_left[0]
height_2 =p_bottom_right[0] - p_top_right[0]

width_1 = p_top_right[1] - p_top_left[1]
width_2 = p_bottom_right[1]- p_bottom_left[1]

max_height=max(int(height_1), int(height_2))
max_width = max(int(width_1), int(width_2))

# four input point 
input_pts = np.float32([[p_top_left[1], p_top_left[0]],
                        [p_top_right[1], p_top_right[0]],
                        [p_bottom_left[1], p_bottom_left[0]],
                        [p_bottom_right[1], p_bottom_right[0]]])

# output points for new transformed image
output_pts = np.float32([[0, 0],
                         [max_width, 0],
                         [0, max_height],
                         [max_width, max_height]])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# Apply the perspective transform to the image
img_gray_oriented = cv2.warpPerspective(img_gray, M, (max_width, max_height))


'''
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(img_gray_oriented),plt.title('Output')
plt.show()
'''



# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray_oriented, (3, 3), 0)

# Apply edge detection (Canny) to find edges in the img
edges = cv2.Canny(blurred, 0, 50)

'''
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Anzahl der Zellen im Gitter
num_cells_x = 13  # Anzahl der Spalten
num_cells_y = 24  # Anzahl der Reihen

# Zellengröße berechnen
cell_width = max_width // num_cells_x
cell_height = max_height // num_cells_y

# Kopie des Bilds zum Zeichnen des Gitters
vertical_lines = []
horizontal_lines = []
min_white_pixels = 1000000000
# Gitterlinien zeichnen
for zoom in range(1,50):
    real_zoom = 1-(zoom * 0.001)
    print("current zoom: ",real_zoom)
    for wiggle_x in range(0,25):
        for wiggle_y in range(0,25):
            grid_image = edges.copy()

            # Listen zur Speicherung der Linienpositionen
            vertical_lines = []
            horizontal_lines = []

            for i in range(1, num_cells_x):
                x = int(i * cell_width * real_zoom) + wiggle_x
                cv2.line(grid_image, (x, 0), (x, int(max_height * real_zoom)), (0, 0, 0), 3)  # Vertikale Linien
                vertical_lines.append(x)

            for j in range(1, num_cells_y):
                y = int(j * cell_height * real_zoom) + wiggle_y
                cv2.line(grid_image, (0, y), (int(max_width * real_zoom), y), (0, 0, 0), 3)  # Horizontale Linien
                horizontal_lines.append(y)

             # Anzahl weißer Pixel zählen
            white_pixels = np.sum(grid_image == 255)

            # Überprüfen, ob dieses Bild weniger weiße Pixel hat
            if white_pixels < min_white_pixels:
                min_white_pixels = white_pixels
                min_white_image = grid_image

#Bild anzeigen
#cv2.imshow('Image with Grid', min_white_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

intersection_points = []
# Finde Schnittpunkte
for x in vertical_lines:
    line_on_x = []
    for y in horizontal_lines:
        line_on_x.append((x, y))
    intersection_points.append(line_on_x)

def analyze_square(img):
    # Optional: Schwellenwert, um Text klarer zu machen
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # OCR ausführen
    custom_config = r'--oem 3 --psm 6'  # Tesseract-Konfiguration
    text = tes.image_to_string(thresh, config=custom_config)

    # Ergebnis analysieren
    if text.strip():  # Prüfen, ob Text gefunden wurde
        print("Text gefunden:")
        print(text)
    else:
        print("Kein Text im Bild gefunden.")
    return 0

wiggle_x = int(img.shape[0]*0.01)
wiggle_y = int(img.shape[1]*0.01)
for x in range(0,num_cells_x-1):
    for y in range (0,num_cells_y-1):
        #get for reference points of square
        top_left = intersection_points[x][y]
        bottom_right = intersection_points[x + 1][y + 1]
        cropped_image = edges[top_left[1]-wiggle_y:bottom_right[1]+wiggle_y, top_left[0]-wiggle_x:bottom_right[0]+wiggle_x]
        analyze_square(cropped_image)





