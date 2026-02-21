# ArUco Drum Coach (CV Prototype)

This repo contains a quick Computer Vision prototype for detecting ArUco markers with OpenCV and drawing a 3D cube + axis on top of each marker. This is the foundation for our drum overlay system.

## What it does
- Opens the camera feed
- Detects ArUco markers (DICT_4X4_50)
- Draws marker outlines + detected IDs
- Estimates pose (rotation + translation) using camera calibration
- Draws a 3D cube that rotates with the marker
- Optionally maps marker IDs to drum labels (SNARE, HIHAT, RIDE)

## Repo layout
- `main.py`  
  Live camera detection + 3D cube overlay
- `calibrate_camera.py`  
  Camera calibration using a chessboard (writes `calib.npz`)
- `calib.npz`  
  Generated camera intrinsics (not committed)
- `marker_*.png`  
  Generated ArUco markers you can print or display

---

## Requirements
- Python 3.9+
- A webcam or camera module
- OpenCV **contrib** build (required for ArUco extras)

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-contrib-python numpy