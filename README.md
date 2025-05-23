# CrosslePyServ

CrosslePyServ is a Python-based server application that processes and analyzes images to create puzzles. It uses computer vision and machine learning techniques to analyze images and convert them into puzzle formats.

## Features

- Image processing and analysis
- Machine learning model for puzzle generation
- Server-client architecture for remote processing
- Training data generation and model learning capabilities
- Real-time status updates during processing

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pytesseract
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CrosslePyServ.git
cd CrosslePyServ
```

2. Install the required dependencies:
```bash
pip install opencv-python numpy matplotlib pytesseract pillow
```

3. Ensure Tesseract OCR is installed on your system:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

The application can be run in several modes:

### Starting the Server
```bash
python main.py --start-server
```

### Running a Test
```bash
python main.py --run-test
```

### Training Data Operations
```bash
# Ensure file paths exist
python main.py --ensure-file-paths

# Generate training data
python main.py --make-training-data

# Format training data
python main.py --format-to-training-data

# Train the model
python main.py --learn
```

## Project Structure

- `main.py`: Main entry point and command-line interface
- `crossle_server.py`: Server implementation for handling client connections
- `analyze.py`: Image analysis and processing logic
- `neuro.py`: Machine learning model implementation
- `puzzle.py`: Puzzle generation and manipulation
- `model/`: Directory containing trained models
- `trainingdata/`: Directory for training data
- `goalcontours/`: Directory containing contour templates

## Server Configuration

The server runs on port 12345 by default. It accepts image data and JSON metadata from clients, processes the images, and returns the results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
