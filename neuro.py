import cv2
import numpy as np
import os

def make_training_data():
    return 0

def generate_arrows(dir):
    # Loop through all files in the directory
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            process_file(file_path)
    return 0

def generate_arrowhandles(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            process_file(file_path)
    return 0

def generate_whites(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            process_file(file_path)
    return 0

