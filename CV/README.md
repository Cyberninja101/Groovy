# ArUco Drum Overlay

Real-time drum trainer/visualizer using OpenCV ArUco pose estimation + 3D projected rings.

## What it does
- Captures camera frames in a background thread for lower latency.
- Detects ArUco markers and tracks corners between detections with optical flow.
- Estimates marker pose and renders 3D rings with `cv2.projectPoints`.
- Supports per-drum 3D ring sizes.
- Supports three modes:
  - `play`: timeline-based pulse playback.
  - `train`: waits for correct hits before advancing.
  - `score`: timeline playback with live accuracy scoring from stick hits.

## Files
- `CV/main.py`: live app (all modes + rendering + stick tracking).
- `CV/calibrate_camera.py`: camera calibration, writes `CV/calib.npz`.
- `CV/calib.npz`: intrinsics/distortion file required for 3D mode.

## Requirements
- Python 3.9+
- `opencv-contrib-python` (for `cv2.aruco`)
- `numpy`
- Camera/webcam

Install:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-contrib-python numpy
```

## Basic run

```bash
python3 CV/main.py
```

Controls:
- `s`: start
- `r`: reset
- `q`: quit
- `n`: skip to next chord (train mode only)

## Modes

### Play mode (default)
Normal timeline pulse visualization (existing behavior).

```bash
python3 CV/main.py --mode play
```

Playback speed:

```bash
python3 CV/main.py --mode play --speed 0.8
python3 CV/main.py --mode play --speed 1.5
```

### Train mode
Song does not advance by time. You must hit expected drum(s) to advance.

```bash
python3 CV/main.py --mode train --stick-track
```

If `--stick-track` is missing, start is blocked and a warning banner is shown.

### Score mode
Song runs on normal timeline while stick hits are matched to expected notes.
Banner shows live accuracy and judgment.

```bash
python3 CV/main.py --mode score --stick-track
```

If `--stick-track` is missing, start is blocked and a warning banner is shown.

Scoring defaults:
- Match window: early `220ms`, late `260ms`.
- Auto timing offset uses EMA (`offset_ms`) and is shown live.
- Judgments:
  - Correct drum: `PERFECT/GOOD/OK/MISS` from timing.
  - Wrong drum in window: partial credit only when close.
- Expected notes with no 3D pose in their window are skipped (not counted in denominator).
- Extra hits are tracked for reporting (not penalized by default).

## Stick tracking flags

```bash
--stick-track
--stick-hsv-lower H,S,V
--stick-hsv-upper H,S,V
--stick-min-area N
--stick-max-jump N
--stick-debug
```

Example:

```bash
python3 CV/main.py --mode score --stick-track \
  --stick-hsv-lower 35,80,80 --stick-hsv-upper 95,255,255 --stick-debug
```

## Raspberry Pi

Use Pi preset:

```bash
python3 CV/main.py --pi --show-fps
```

More speed (lower quality):

```bash
python3 CV/main.py --pi --detect-scale 0.6 --detect-every 3 --show-fps
```

More quality (higher CPU):

```bash
python3 CV/main.py --pi --detect-scale 0.8 --detect-every 1 --show-fps
```

## Per-drum 3D size

Set per-drum radii in `make_config()` (`drum_name_to_highlight_r_m`).

These affect:
- base 3D ring size
- pulse animation scale/thickness
- roll animation scale/thickness
