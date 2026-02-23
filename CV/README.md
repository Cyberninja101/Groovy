# Groovy CV Runtime

Realtime camera engine for marker tracking, ring rendering, stick tracking, and gameplay.

## What It Does
- Captures frames in a background thread for low-latency processing.
- Detects ArUco markers and estimates marker pose in 3D.
- Renders projected drum rings and pulse animations.
- Tracks up to two green stick tips (`T1`, `T2`) in debug mode.
- Runs `play`, `train`, and `score` modes with live UI feedback.
- Runs fullscreen output by default (`drum_overlay` window).

## Core Files
- `CV/main.py`: main runtime and all gameplay/render logic.
- `CV/calibrate_camera.py`: generates camera intrinsics file.
- `CV/calib.npz`: calibration file required for 3D pose/ring rendering.

## Requirements
- Python 3.9+
- `opencv-contrib-python`
- `numpy`
- webcam (or Continuity Camera/iPhone camera)

Install:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-contrib-python numpy
```

## Run

```bash
python3 CV/main.py --mode score --stick-track
```

Common options:

```bash
--mode {play,train,score}
--speed FLOAT
--stick-track
--stick-hsv-lower H,S,V
--stick-hsv-upper H,S,V
--stick-min-area N
--stick-max-jump N
--stick-debug
--auto-start-delay SECONDS
--pi
--show-fps
```

Controls:
- `s` start
- `r` reset
- `q` or `Esc` quit
- `n` skip to next chord (`train` mode only)

## Modes

### Play
- Timeline pulse visualization + hit flash feedback.
- No scoring judgment pipeline.

```bash
python3 CV/main.py --mode play --speed 1.0
```

### Train
- Time does not advance by timeline.
- You must hit expected drum(s) to move forward.
- Start is blocked unless `--stick-track` is enabled.

```bash
python3 CV/main.py --mode train --stick-track
```

### Score
- Timeline advances with optional speed scaling.
- Expected notes are matched to stick hits.
- Live banner shows accuracy, counts, and offset.
- Every 10 scored notes (including misses) triggers a center popup:
  `Last 10 acc: XX% (...)`.

```bash
python3 CV/main.py --mode score --stick-track --stick-debug
```

## Stick Tracking Notes
- Stick detection is HSV-based; default range is green-ish:
  - lower: `35,80,80`
  - upper: `95,255,255`
- Hit detection is intentionally forgiving and prioritizes being inside the drum area over speed.
- `--stick-debug` opens `stick_mask` and shows tip overlays (`T1`, `T2`) plus nearest-marker telemetry.

## Startup Countdown
- If `--auto-start-delay` is set, a large center-screen countdown is shown until start.

Example:

```bash
python3 CV/main.py --mode score --stick-track --auto-start-delay 3 --stick-debug
```

## Raspberry Pi Preset
- `--pi` adjusts detection/render defaults for lower compute devices.

```bash
python3 CV/main.py --pi --mode score --stick-track --show-fps
```

## Drum Circle Size
Edit per-drum radii in `make_config()` (`drum_name_to_highlight_r_m`).

Higher number means bigger ring (radius in meters on marker plane).

## Default Song Source
`Config.song_path` currently points to:

`App/backend/beatmaps/seven_nation_army.json`
 