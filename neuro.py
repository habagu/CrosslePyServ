import cv2
import numpy as np
import os

def make_training_data():
    dirpre = "./trainigdata/base/"
    traget_dirpre = "./trainigdata/generated/"
    post = "white/"
    whites_path = traget_dirpre + post
    generate_whites(dirpre + post,traget_dirpre + post)
    post = "arrow/"
    generate_arrows(dirpre + post, whites_path, traget_dirpre + post)
    post = "handle/top_to_right/"
    generate_arrowhandles(dirpre + post, whites_path)
    post = "handle/bottom_to_right/"
    generate_mirror_arrowhandles(dirpre + post, whites_path)
    post = "double_arrow/"
    generate_double_arrow(dirpre + post, whites_path, traget_dirpre + post)
    generate_double_arrow_from_arrows(dirpre + "arrow", whites_path, traget_dirpre + post)
    return 0

def generate_double_arrow_from_arrows(dir, whites_dir, target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            to_right = cv2.imread(file_path)

            for filename2 in os.listdir(dir):
                file_path2 = os.path.join(dir, filename2)
                
                # Ensure it's a file (not a directory)
                if os.path.isfile(file_path2):
                    to_bottom = cv2.imread(file_path)
                    image = cv2.addWeighted(to_bottom, 0.5,to_right, 0.5, 0)
                    image[image <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir),image)
    return 0

def generate_double_arrow(dir, whites_dir, target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir), image)
    return 0

def generate_arrows(dir, whites_dir, target_dir):
    # Loop through all files in the directory
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir + "/to_right"), image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir + "/to_bottom"), rotated_image)
    return 0

def generate_mirror_arrowhandles(dir, whites_dir, target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        if os.path.isfile(file_path):
            image = cv2.flip(cv2.imread(file_path),0)
            generate_arrowhandles_core(image, whites_dir, target_dir)
    return 0

def generate_arrowhandles(dir, whites_dir, target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            generate_arrowhandles_core(image, whites_dir, target_dir)
    return 0

def generate_arrowhandles_core(image,whites_dir, target_dir):
    cv2.imwrite(name_in_target(target_dir), image)
    add_whites(image,whites_dir,target_dir)

    cv2.imwrite(name_in_target(target_dir),cv2.flip(image, 0))
    add_whites(cv2.flip(image, 0),whites_dir,target_dir)
    # Bild um 90° nach rechts drehen
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Gedrehtes Bild speichern
    cv2.imwrite(name_in_target(target_dir), rotated_image)
    add_whites(rotated_image,whites_dir,target_dir)
    
    cv2.imwrite(name_in_target(target_dir),cv2.flip(rotated_image, 0))
    add_whites(cv2.flip(rotated_image, 0),whites_dir,target_dir)
    return 0

def add_whites(image,whites_dir,target_dir):
    for whitename in os.listdir(whites_dir):
        white_path = os.path.join(whites_dir, whitename)

        if os.path.isfile(white_path):
            for i in range(0,4):
                if i == 0 :
                    white = cv2.imread(white_path)   
                elif i == 1:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 2:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 3:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)

                for j in range(-1,2):
                    white = cv2.flip(white, j)
                    fused = cv2.addWeighted(white, 0.5,image, 0.5, 0)
                    fused[fused <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir), fused)
    return 0

def generate_whites(dir, target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir) , image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

    return 0

def count_files(dir):
    return len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

def name_in_target(dir):
    n = count_files(dir)
    return dir + str(n+1)