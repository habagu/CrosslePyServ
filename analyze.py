#!/usr/bin/env python

import math
from crossle_server import send_status_updates,send_response
from neuro import predict,delete_png_files,progress_print
from puzzle import Puzzle_to_JsonArray
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract as tes

def analyze_image(path: str = "./sample_points.jpg", json_data: dict = None, client_socket = None):
    send_status_updates("Starting analyzing image...", 0, client_socket)
    
    img_src = cv2.imread(path)

    # Anzahl der Zellen im Gitter
    num_cells_x = 13  # Anzahl der Spalten
    num_cells_y = 24  # Anzahl der Reihen

    #correct image oreientation
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    # loop over the contours add possible center points to point array
    pts = []
    if json_data is not None:
        num_cells_x = json_data["columns"]
        num_cells_y = json_data["rows"]
        pts = json_data["points"]
    else:
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

    # 2. (Optional) Kontrastanpassung mit CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(img_gray_oriented)

    # 3. Optimierte Schwellenwerte und L2-Norm in Canny anwenden
    edges = cv2.Canny(contrast_enhanced, 30, 60, apertureSize=3, L2gradient=True)


    # 4. (Optional) Morphologische Nachbearbeitung (Dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

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
        progress_print(str(percentage_to_finish) + "%")
        send_status_updates("fiting grid: ",percentage_to_finish,client_socket)
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

    def analyze_square(img,x,y):

        # Load the image in grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the mean pixel value
        mean_pixel_value = np.mean(img_gray)

        # Apply thresholding using the mean value
        _, binary_img = cv2.threshold(img_gray, mean_pixel_value, 255, cv2.THRESH_BINARY)

        # Create a copy of the image to modify
        modified_image = binary_img.copy()

        # Define the black pixel threshold (95%)
        black_threshold_col = 0.90 * binary_img.shape[0]  # 95% of the rows
        black_threshold_row = 0.90 * binary_img.shape[1]  # 95% of the columns

        # Iterate through each column
        for col in range(binary_img.shape[1]):  # image.shape[1] is the number of columns (width)
            black_pixel_count = np.sum(binary_img[:, col] == 0)  # Count black pixels in the column
            if black_pixel_count >= black_threshold_col:  # Check if >= 95% are black
                modified_image[:, col] = 255  # Set the entire column to white

        # Iterate through each row
        for row in range(binary_img.shape[0]):  # image.shape[0] is the number of rows (height)
            black_pixel_count = np.sum(binary_img[row, :] == 0)  # Count black pixels in the row
            if black_pixel_count >= black_threshold_row:  # Check if >= 95% are black
                modified_image[row, :] = 255  # Set the entire row to white

        # Convert the binary image to RGB
        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2RGB)
        pathpre = "./sol/"
        pathpost = str(x) +"_" +str(y) + ".png"
        path = pathpre + pathpost
        delete_png_files(pathpre)
        field = {
                "x":x,"y":y,
                "solution":False,
                "white":False,
                "arrow_to_bottom":False,
                "arrow_to_right":False,
                "double_arrow":False,
                "handle_bottom_to_right":False,
                "handle_left_to_bottom":False,
                "handle_right_to_bottom":False,
                "handle_top_to_right":False,
                "text":False,
                "text_value":""}
        if check_for_solution_field(modified_image):
            cv2.imwrite(path + "sol",modified_image)
            field["solution"] = True

        white = 0
        arrow_to_bottom = 1
        arrow_to_right = 2
        double_arrow = 3
        handle_bottom_to_right = 4
        handle_left_to_bottom = 5
        handle_right_to_bottom = 6
        handle_top_to_right = 7
        prediction = predict(modified_image)
        print(prediction)
        if not check_for_center_white(modified_image):
            field["text"] = True
            field["text_value"] = check_for_text(modified_image)
            cv2.imwrite(path + "text" + pathpost,modified_image)
        elif prediction == white:
            cv2.imwrite(path + "white" + pathpost,modified_image)
            field["white"] = True
        elif prediction == arrow_to_bottom:
            cv2.imwrite(path + "arrow_to_bottom" + pathpost,modified_image)
            field["arrow_to_bottom"] = True
        elif prediction == arrow_to_right:
            cv2.imwrite(path + "arrow_to_right" + pathpost,modified_image)
            field["arrow_to_right"] = True
        elif prediction == double_arrow:
            cv2.imwrite(path + "double_arrow" + pathpost,modified_image)
            field["double_arrow"] = True
        elif prediction == handle_bottom_to_right:
            cv2.imwrite(path + "handle_bottom_to_right" + pathpost,modified_image)
            field["handle_bottom_to_right"] = True
        elif prediction == handle_left_to_bottom:
            cv2.imwrite(path + "handle_left_to_bottom" + pathpost,modified_image)
            field["handle_left_to_bottom"] = True
        elif prediction == handle_right_to_bottom:
            cv2.imwrite(path + "handle_right_to_bottom" + pathpost,modified_image)
            field["handle_right_to_bottom"] = True
        elif prediction == handle_top_to_right:
            cv2.imwrite(path + "handle_top_to_right" + pathpost,modified_image)
            field["handle_top_to_right"] = True
        return field

    def make_binary(img):
        # Load the image in grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the mean pixel value
        mean_pixel_value = np.mean(img_gray)

        # Apply thresholding using the mean value
        _, binary_img = cv2.threshold(img_gray, mean_pixel_value, 255, cv2.THRESH_BINARY)

        # Create a copy of the image to modify
        return binary_img

    def check_for_solution_field(img):
        path = "./goalcontours/SolutionField.png"
        arrow_src = cv2.imread(path)

        # Ensure both images are the same size
        arrow_src = cv2.resize(arrow_src, (img.shape[1], img.shape[0]))

        # Convert black pixels in the black image to red
        arrow_src[(arrow_src == 0).all(axis=-1)] = [0, 0, 255]
        red_arrow = arrow_src  # Set black pixels to red

        # Overlay the red image onto the base image
        overlaid_image = cv2.addWeighted(red_arrow, 1,img, 1, 0)
        
        # Check for red pixels in the overlaid image
        red_pixel_count_og = np.sum(np.all(red_arrow == [0, 0, 255], axis=-1))
        red_pixel_count_overlaid = np.sum(np.all(overlaid_image == [0, 0, 255], axis=-1))
        red_pixel_percentage = red_pixel_count_overlaid / red_pixel_count_og * 100

        # Count the number of black pixels (value 0)
        black_pixels = np.sum(img == 0)
        black_pixels_percentage = black_pixels/(img.shape[0]*img.shape[1]) * 100
        if red_pixel_percentage > 13.8 and black_pixels_percentage < 50 and black_pixels_percentage > 28:
            return True
        else:
            return False

    def check_for_arrow(img):

        return False

    def mirror_image_horizontal(image):
        # Get the image dimensions
        return cv2.flip(image, 0)

    def rotate_image_by_case(image, case):
        # Define the rotation cases
        if case == 0:
            rotated_image = image  # 0° rotation (no change)
        elif case == 1:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 90° clockwise
        elif case == 2:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)  # 180°
        elif case == 3:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270° (90° counterclockwise)
        else:
            raise ValueError("Invalid case. Case must be 0, 1, 2, or 3.")
        
        return rotated_image

    def check_for_center_white(img):
        height_range=(0.4, 0.6)
        width_range=(0.2, 0.8)
        # Get image dimensions
        height, width = img.shape[:2]

        # Calculate pixel ranges for cropping
        y1 = int(height * height_range[0])
        y2 = int(height * height_range[1])
        x1 = int(width * width_range[0])
        x2 = int(width * width_range[1])

        # Crop the image
        cropped_image = img[y1:y2, x1:x2]    

        # Total number of pixels
        total_pixels = cropped_image.size

    
        black_pixels = np.sum(img == 0)
        black_pixels_percentage = black_pixels/(img.shape[0]*img.shape[1]) * 100
        if black_pixels_percentage < 20:
            return True
        else:
            # Count the number of black pixels (value 0)
            black_pixels = np.sum(cropped_image == 0)

            # Calculate the percentage of black pixels
            black_pixel_percentage = (black_pixels / total_pixels) * 100

            if black_pixel_percentage < 10:
                return True
            else:
                return False

    def check_for_text(img): # OCR ausführen
        custom_config = r'--oem 3 --psm 6'  # Tesseract-Konfiguration
        return tes.image_to_string(img, config=custom_config)


    zoom_x = int(img_src.shape[0]*0.002)
    zoom_y = int(img_src.shape[1]*0.002)
    img_for_cropping = img_src_oriented.copy()

    n=0
    Puzzle = []
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
            percentage_to_finish = percentage_to_finish + 50/((len(intersection_points)-1)*(len(intersection_points[x])-1))
            Puzzle.append(analyze_square(cropped_image,x,y))
            progress_print(str(percentage_to_finish) + "%")
            send_status_updates("Analysing image: ",percentage_to_finish,client_socket)

    JsonArray = Puzzle_to_JsonArray(Puzzle)
    solutions = [e for e in Puzzle if e.get("sol") is True]
    solutionsArray = []
    for sol in solutions:
        solutionsArray.append(str(sol.get("x")) + ":" + str(sol.get("y")))
    status = {
        "type": "finished",
        "data": {
            "rows": num_cells_y,
            "columns": num_cells_x,
            "puzzle": {
                "questions": JsonArray,
                "solutions": solutionsArray
            }
        }
    }
    
    send_response(client_socket, status)
    return 0
