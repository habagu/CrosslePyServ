#!/usr/bin/env python

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract as tes
from neuro import ensurefilepaths,make_training_data,format_to_training_data_and_validat_data,learn
from crossle_server import initialize_server,test_image
import sys
import argparse

def __main__():
    parser = argparse.ArgumentParser(description='Analyze an image and convert it to a puzzle.')
    parser.add_argument('--start-server', action='store_true', help='Start the server')
    parser.add_argument('--run-test', action='store_true', help='Run a test with default values')
    parser.add_argument('--ensure-file-paths', action='store_true', help='Ensure that all file paths exist')
    parser.add_argument('--make-training-data', action='store_true', help='Make training data')
    parser.add_argument('--format-to-training-data', action='store_true', help='Format the training data')
    parser.add_argument('--learn', action='store_true', help='Learn')
    args = parser.parse_args()

    if args.start_server:
        initialize_server()
    elif args.run_test:
        test_image()
    else:
        if args.ensure_file_paths:
            ensurefilepaths()
        if args.make_training_data:
            make_training_data()
        if args.format_to_training_data:
            format_to_training_data_and_validat_data()
        if args.learn:
            learn()
        if not (args.start_server or args.run_test or args.ensure_file_paths or args.make_training_data or args.format_to_training_data or args.learn):
            parser.print_help()

    return 0

if __name__ == "__main__":
    __main__()