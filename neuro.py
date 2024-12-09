import cv2
import numpy as np
import os

def make_training_data():
    dirpre = "./trainigdata/base"
    traget_dirpre = "./trainigdata/generated/"
    post = "arrow"
    generate_arrows(dirpre + post,traget_dirpre + post)
    post = "handle"
    generate_arrowhandles(dirpre + post,traget_dirpre + post)
    post = "white"
    generate_whites(dirpre + post,traget_dirpre + post)
    return 0

def generate_arrows(dir,target_dir):
    # Loop through all files in the directory
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(target_dir, image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(target_dir, rotated_image)
    return 0

def generate_arrowhandles(dir,target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(target_dir, image)

            cv2.imwrite(target_dir,cv2.flip(image, 0))

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(target_dir, rotated_image)
            cv2.imwrite(target_dir,cv2.flip(rotated_image, 0))
    return 0

def generate_whites(dir,target_dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(target_dir, image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(target_dir, rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(target_dir, rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(target_dir, rotated_image)

    return 0

