#!/usr/bin/env python

from crossle_server import start_server
import sys
from analyze import analyze_image
import os

def main():
    mode = input("Enter mode (debug/server): ").strip().lower()
    os.system('cls' if os.name == 'nt' else 'clear')
    if mode == 'debug':
        analyze_image()
    else:
        start_server()

if __name__ == "__main__":
    main()