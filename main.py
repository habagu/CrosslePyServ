#!/usr/bin/env python

import math
import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract as tes
from langdetect import detect, detect_langs

path = "/home/gntjonau/crossle/CrosslePyServ/sample_points.jpg"

img_src = cv2.imread(path)

#correct image oreientation
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

# Define the HSV range for green
lower_green = np.array([35, 100, 100])  # Lower bound of green
upper_green = np.array([85, 255, 255])  # Upper bound of green
mask = cv2.inRange(hsv,lower_green,upper_green)
res = cv2.bitwise_and(img_src,img_src,mask=mask)
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
        cv2.drawContours(img_src,cnt,-1,(60,255,255),4)

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
    return math.sqrt((p[0] - 0)**2 + (p[1] - img_src.shape[1])**2)
def distance_bottom_left (p): 
    return math.sqrt((p[0] - img_src.shape[0])**2 + (p[1] - 0)**2)
def distance_bottom_right (p): 
    return math.sqrt((p[0] - img_src.shape[0])**2 + (p[1] - img_src.shape[1])**2)

        


p_top_left = min(pts, key = distance_top_left)
p_top_right = min(pts, key = distance_top_right)
p_bottom_left = min(pts, key = distance_bottom_left)
p_bottom_right = min(pts, key = distance_bottom_right)

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

img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# Apply the perspective transform to the image
img_gray_oriented = cv2.warpPerspective(img_gray, M, (max_width, max_height))
img_src_oriented = cv2.warpPerspective(img_src, M, (max_width, max_height))


'''
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(img_gray_oriented),plt.title('Output')
plt.show()
'''



# 2. (Optional) Kontrastanpassung mit CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(img_gray_oriented)

# 3. Optimierte Schwellenwerte und L2-Norm in Canny anwenden
edges = cv2.Canny(contrast_enhanced, 30, 60, apertureSize=3, L2gradient=True)


# 4. (Optional) Morphologische Nachbearbeitung (Dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

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
vertical_lines_best = []
horizontal_lines_best = []
min_white_pixels = 1000000000
percentage_to_finish = 0
# Gitterlinien zeichnen
for zoom in range(0,50):
    real_zoom = 1-(zoom * 0.001)
    percentage_to_finish = percentage_to_finish + 1
    print(percentage_to_finish,"%")
    for wiggle_x in range(0,13):
        for wiggle_y in range(0,13):
            grid_image = edges.copy()

            # Listen zur Speicherung der Linienpositionen
            vertical_lines = []
            horizontal_lines = []

            for i in range(0, num_cells_x+1):
                x = int(i * cell_width * real_zoom) + wiggle_x
                cv2.line(grid_image, (x, 0), (x, int(max_height * real_zoom)), (0, 125, 0), 2)  # Vertikale Linien
                vertical_lines.append(x)

            for j in range(0, num_cells_y+1):
                y = int(j * cell_height * real_zoom) + wiggle_y
                cv2.line(grid_image, (0, y), (int(max_width * real_zoom), y), (0, 125, 0), 2)  # Horizontale Linien
                horizontal_lines.append(y)

             # Anzahl weißer Pixel zählen
            white_pixels = np.sum(grid_image == 255)

            # Überprüfen, ob dieses Bild weniger weiße Pixel hat
            if white_pixels < min_white_pixels:
                min_white_pixels = white_pixels
                min_white_image = grid_image
                vertical_lines_best = vertical_lines
                horizontal_lines_best = horizontal_lines

intersection_points = []
# Finde Schnittpunkte
for x in vertical_lines_best:
    line_on_x = []
    for y in horizontal_lines_best:
        line_on_x.append((x, y))
    intersection_points.append(line_on_x)

def analyze_square(img):

    # OCR ausführen
    custom_config = r'--oem 3 --psm 6'  # Tesseract-Konfiguration
    text = tes.image_to_string(img, config=custom_config)
    if len(text)>0:  # Prüfen, ob Text gefunden wurde
        lang = ''
        try:
            langs = detect_langs(text)
            print(langs)
            if "de" in langs:
                print("Text gefunden:")
                print(text)
                return 1
            else:
                no_text(img)
        except:
            no_text(img)
        
    return 0

def no_text(img):
    print("Kein Text")

zoom_x = int(img_src.shape[0]*0.002)
zoom_y = int(img_src.shape[1]*0.002)
img_for_cropping = img_src_oriented.copy()

n=0
for x in range(0,len(intersection_points)-1):
    for y in range (0,len(intersection_points[x])-1):


        #get for reference points of square
        top_left = intersection_points[x][y]
        bottom_right = intersection_points[x + 1][y + 1]

        x1 = top_left[0]-zoom_x
        x2 = bottom_right[0]+zoom_x
        y1 = top_left[1]-zoom_y
        y2 = bottom_right[1]+zoom_y

        if x1<0:
            x1=0
        if x2>img_for_cropping.shape[1]:
            x2 = img_for_cropping.shape[1]
        if y1<0:
            y1=0
        if y2>img_for_cropping.shape[0]:
            y2 = img_for_cropping.shape[0]

        cropped_image = img_for_cropping[y1:y2, x1:x2]
        print(x,y)
        n = n + analyze_square(cropped_image)

        percentage_to_finish = percentage_to_finish + 50/((len(intersection_points)-1)*(len(intersection_points[x])-1))
        print(percentage_to_finish,"%")

print(n, "should be 78")
            





