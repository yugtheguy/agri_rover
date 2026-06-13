# Rover Weed Detection System

This repository contains a computer vision pipeline for real-time weed detection using YOLOv8. It is designed to run inference both on static images and on a live video stream from an ESP32 camera, identifying "crop" versus "weed" to potentially assist a rover in agricultural tasks.

## Directory Structure

- `data/` - Contains the dataset organized into `train`, `valid`, and `test` splits.
- `scripts/` - Python scripts for training, testing, validation, and inference.
- `requirements.txt` - Project dependencies.

## Key Features

- **Object Detection Model:** Utilizes [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (nano model: `yolov8n.pt`) for fast and efficient real-time detection.
- **Classes:** The model is trained to differentiate between:
  - `0: crop`
  - `1: weed`
- **Live Stream Inference:** Integrates with an ESP32 camera stream to perform live detection and save frames containing weeds.

## Installation

Ensure you have Python 3 installed. You can set up a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

### Dependencies
- `ultralytics>=8.0.0`
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`

## Scripts Overview

All scripts are located in the `scripts/` directory:

- **`train.py`**: Trains the YOLOv8 model using the dataset specified in `data.yaml`. The model is trained for 25 epochs with a batch size of 8.
- **`inference.py`**: Runs inference on a single static image provided via command-line arguments and displays the results.
  ```bash
  python scripts/inference.py path/to/image.jpg
  ```
- **`live_esp32.py`**: Connects to an ESP32 camera stream (default URL: `http://192.168.1.211:81/stream`), performs real-time weed detection, and saves any frames containing weeds into a `weed_detected_frames` directory.
- **`test_accuracy.py`**: Evaluates the trained model against the test dataset and calculates metrics like True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), Accuracy, Precision, and Recall.
- **`validate_dataset.py`**: Utility script to validate the YOLO annotation format of the training dataset. It checks for missing labels/images, invalid coordinates, invalid class IDs, and outputs the class distribution.
- **`test_setup.py`**: A quick setup validation script to check if PyTorch, CUDA, and Ultralytics YOLO are correctly loaded.

## Dataset Structure

The project expects a standard YOLO dataset structure defined in `scripts/data.yaml`:

```yaml
path: ../data
train: train/images
val: valid/images
test: test/images
names:
  0: crop
  1: weed
```

## How to Train

1. Ensure your data is placed in the correct directories as defined by `data.yaml`.
2. Run the training script:
   ```bash
   python scripts/train.py
   ```
3. The trained weights will be saved in the `runs/` directory (e.g., `runs/weed_model/weights/best.pt`).
