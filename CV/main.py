import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    # Song
    song_path: str = "Audio/brandy-1350.json"

    # Song uses numeric drum IDs, map those to human names
    drum_id_to_name: Dict[int, str] = field(default_factory=dict)

    # Map drum name -> marker id (ArUco ID on that drum)
    name_to_marker_id: Dict[str, int] = field(default_factory=dict)

    # Map drum name -> 3D ring radius (meters)
    drum_name_to_highlight_r_m: Dict[str, float] = field(default_factory=dict)

    # Rhythm-game animation
    pulse_min_scale: float = 0.55
    pulse_max_scale: float = 2.50
    hit_flash_ms: int = 40
    hit_fill_alpha: float = 0.30

    # Timing behavior
    lead_ms: int = 350
    hold_ms: int = 120
    chord_eps_ms: int = 50

    # Window + queue behavior
    lookahead_ms: int = 300          # pre-ingest events this far beyond lead
    roll_merge_ms: int = 150         # merge rapid repeats on same drum into a roll
    max_pulses_per_marker: int = 10  # safety cap
    cleanup_margin_ms: int = 60      # extra buffer before deleting expired pulses

    # 3D / Calibration
    use_3d: bool = True
    calib_path: str = "CV/calib.npz"
    marker_length_m: float = 0.05
    drum_highlight_r_m: float = 0.14
    circle_pts: int = 64
    min_circle_pts: int = 16
    pose_smooth_alpha: float = 0.55

    # Video
    cam_index: int = 0
    frame_w: int = 960
    frame_h: int = 540

    # Visual
    base_ring_thickness: int = 12
    base_ring_alpha: float = 0.6

    # Performance tuning
    detect_downscale: float = 1.0          # detect on smaller image, then scale corners back
    detect_every_n_frames: int = 1         # reuse last detection on skipped frames
    max_cached_detection_frames: int = 1   # how long cached detection is allowed to persist
    corner_refine: bool = True
    show_fps: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco drum overlay")
    parser.add_argument("--pi", action="store_true", help="Enable Raspberry Pi performance preset.")
    parser.add_argument("--detect-scale", type=float, default=None, help="Marker detection scale in (0, 1].")
    parser.add_argument("--detect-every", type=int, default=None, help="Run marker detection every N frames (>= 1).")
    parser.add_argument("--show-fps", action="store_true", help="Show realtime FPS in the overlay.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        drum_id_to_name={
            3: "SNARE",
            1: "BASS",
            5: "HIHAT",
            6: "CRASH",
        },
        name_to_marker_id={
            "SNARE": 1,
            "BASS": 2,
            "RIDE": 3,
            "CRASH": 4,
        },
        drum_name_to_highlight_r_m={
            "SNARE": 0.14,
            "BASS": 0.20,
            "RIDE": 0.17,
            "CRASH": 0.16,
        },
    )

    if args.pi:
        cfg = replace(
            cfg,
            frame_w=640,
            frame_h=360,
            use_3d=True,
            circle_pts=32,
            min_circle_pts=12,
            pose_smooth_alpha=0.50,
            base_ring_thickness=8,
            detect_downscale=0.7,
            detect_every_n_frames=2,
            max_cached_detection_frames=1,
            corner_refine=False,
        )

    if args.detect_scale is not None:
        cfg = replace(cfg, detect_downscale=max(0.25, min(1.0, float(args.detect_scale))))
    if args.detect_every is not None:
        cfg = replace(cfg, detect_every_n_frames=max(1, int(args.detect_every)))
    if args.show_fps:
        cfg = replace(cfg, show_fps=True)

    cfg = replace(
        cfg,
        use_3d=True,  # 3D-only rendering path
        max_cached_detection_frames=max(cfg.max_cached_detection_frames, cfg.detect_every_n_frames - 1),
    )

    return cfg


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Pulse:
    """
    kind:
      - 'hit': single hit at t_hit_ms
      - 'roll': merged rapid repeats from t_hit_ms .. t_end_ms
    """
    kind: str
    t_hit_ms: int
    t_end_ms: Optional[int] = None   # only for roll
    count: int = 1                   # only for roll (approx intensity)
    last_hit_ms: Optional[int] = None  # for roll flash timing


# ============================================================
# HELPERS
# ============================================================

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def smoothstep01(x: float) -> float:
    t = clamp01(x)
    return t * t * (3.0 - 2.0 * t)


def bbox_from_corners(corners_1x4x2: np.ndarray) -> Tuple[int, int, int, int]:
    pts = corners_1x4x2.reshape(-1, 2)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def make_marker_radius_map(cfg: Config) -> Dict[int, float]:
    marker_radii: Dict[int, float] = {}
    for name, mid in cfg.name_to_marker_id.items():
        r = float(cfg.drum_name_to_highlight_r_m.get(name, cfg.drum_highlight_r_m))
        if (not np.isfinite(r)) or (r <= 0.0):
            r = float(cfg.drum_highlight_r_m)
        marker_radii[int(mid)] = r
    return marker_radii


def smooth_pose(
    prev_pose: Tuple[np.ndarray, np.ndarray],
    curr_pose: Tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    a = clamp01(float(alpha))
    prev_r, prev_t = prev_pose
    curr_r, curr_t = curr_pose

    prev_r = np.asarray(prev_r, dtype=np.float64).reshape(3, 1)
    prev_t = np.asarray(prev_t, dtype=np.float64).reshape(3, 1)
    curr_r = np.asarray(curr_r, dtype=np.float64).reshape(3, 1)
    curr_t = np.asarray(curr_t, dtype=np.float64).reshape(3, 1)

    # Keep axis-angle direction consistent to avoid jitter from sign flips.
    if float(np.dot(prev_r.reshape(3), curr_r.reshape(3))) < 0.0:
        curr_r = -curr_r

    out_r = (1.0 - a) * prev_r + a * curr_r
    out_t = (1.0 - a) * prev_t + a * curr_t
    return out_r, out_t


def pick_project_circle_pts(
    mtx: np.ndarray,
    tvec: np.ndarray,
    radius_m: float,
    max_pts: int,
    min_pts: int,
) -> int:
    max_p = max(8, int(max_pts))
    min_p = min(max_p, max(8, int(min_pts)))
    tz = float(np.asarray(tvec, dtype=np.float64).reshape(3, 1)[2, 0])
    if tz <= 1e-5:
        return max_p

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    f = 0.5 * (fx + fy)
    if not np.isfinite(f) or f <= 1e-5:
        return max_p

    px_radius = abs(f * float(radius_m) / tz)
    target = int(round(0.60 * px_radius))
    return int(np.clip(target, min_p, max_p))


def safe_load_calibration(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        calib = np.load(path)
        mtx = calib["mtx"]
        dist = calib["dist"]
        if not (np.isfinite(mtx).all() and np.isfinite(dist).all()):
            print("Warning: calibration contains NaN/Inf.")
            return None, None
        return mtx, dist
    except Exception as e:
        print(f"Warning: could not load calibration '{path}'. Error: {e}")
        return None, None


# ============================================================
# SONG LOADING + TIMING
# ============================================================

def load_song(path: str) -> Tuple[str, List[dict]]:
    with open(path, "r") as f:
        data = json.load(f)
    events = list(data.get("events", []))
    events.sort(key=lambda e: e["t_ms"])
    return data.get("title", "song"), events


def advance_idx_past_old(events: List[dict], idx: int, now_ms: int, hold_ms: int) -> int:
    while idx < len(events) and events[idx]["t_ms"] < now_ms - hold_ms:
        idx += 1
    return idx


def group_chord(events: List[dict], idx: int, chord_eps_ms: int) -> Tuple[int, List[dict]]:
    t0 = events[idx]["t_ms"]
    group = [events[idx]]
    j = idx + 1
    while j < len(events) and abs(events[j]["t_ms"] - t0) <= chord_eps_ms:
        group.append(events[j])
        j += 1
    return t0, group


# ============================================================
# DRAWING
# ============================================================

def _clip_roi(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    xi0 = max(0, int(np.floor(x0)))
    yi0 = max(0, int(np.floor(y0)))
    xi1 = min(w, int(np.ceil(x1)))
    yi1 = min(h, int(np.ceil(y1)))
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    return xi0, yi0, xi1, yi1


def alpha_fill_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    a = float(clamp01(alpha))
    if a <= 0:
        return
    roi_box = _clip_roi(frame, x0, y0, x1, y1)
    if roi_box is None:
        return
    rx0, ry0, rx1, ry1 = roi_box
    roi = frame[ry0:ry1, rx0:rx1]
    if a >= 0.995:
        roi[:] = color
        return
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, a, roi, 1.0 - a, 0, dst=roi)


def alpha_draw_circle(
    frame: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    if r <= 0:
        return
    a = float(clamp01(alpha))
    if a <= 0:
        return
    if a >= 0.995:
        cv2.circle(frame, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
        return

    pad = max(2, abs(thickness) + 2)
    roi_box = _clip_roi(frame, cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad)
    if roi_box is None:
        return
    x0, y0, x1, y1 = roi_box
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.circle(overlay, (cx - x0, cy - y0), r, color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, a, roi, 1.0 - a, 0, dst=roi)


def alpha_draw_polyline(
    frame: np.ndarray,
    poly: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    if poly.size == 0:
        return
    a = float(clamp01(alpha))
    if a <= 0:
        return
    if a >= 0.995:
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return

    pad = max(2, thickness + 2)
    pts = poly.reshape(-1, 2).astype(np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)

    roi_box = _clip_roi(frame, min_xy[0] - pad, min_xy[1] - pad, max_xy[0] + pad, max_xy[1] + pad)
    if roi_box is None:
        return

    x0, y0, x1, y1 = roi_box
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    shifted = poly.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    cv2.polylines(overlay, [shifted], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, a, roi, 1.0 - a, 0, dst=roi)


def alpha_fill_polygon(
    frame: np.ndarray,
    poly: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    if poly.size == 0:
        return
    a = float(clamp01(alpha))
    if a <= 0:
        return
    if a >= 0.995:
        cv2.fillPoly(frame, [poly], color, lineType=cv2.LINE_AA)
        return

    pts = poly.reshape(-1, 2).astype(np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    pad = 2
    roi_box = _clip_roi(frame, min_xy[0] - pad, min_xy[1] - pad, max_xy[0] + pad, max_xy[1] + pad)
    if roi_box is None:
        return

    x0, y0, x1, y1 = roi_box
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    shifted = poly.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    cv2.fillPoly(overlay, [shifted], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, a, roi, 1.0 - a, 0, dst=roi)


def draw_banner(frame: np.ndarray, text: str, ok: bool = True) -> None:
    _h, w = frame.shape[:2]
    bar_h = 60
    alpha_fill_rect(frame, 0, 0, w, bar_h, (0, 0, 0), 0.55)

    color = (0, 255, 0) if ok else (0, 255, 255)
    cv2.putText(frame, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def draw_2d_ring(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color=(0, 255, 0),
    alpha: float = 0.6,
    thickness: int = 12,
    radius_scale: float = 1.0,
) -> None:
    cx, cy = x + w // 2, y + h // 2
    base_r = 0.60 * max(w, h)
    r = int(base_r * radius_scale)
    alpha_draw_circle(frame, cx, cy, r, color, thickness, alpha)


def draw_projected_ring_safe(
    frame: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    radius_m: float,
    circle_pts: int,
    min_circle_pts: int = 16,
    color=(0, 0, 255),
    alpha: float = 0.25,
    thickness: int = 6,
    fill_alpha: float = 0.0,
) -> bool:
    """
    Projects a ring on the marker plane using its pose.
    Returns True if drawn, False if pose/projection is unstable (NaN/Inf or behind camera).
    """
    if mtx is None or dist is None:
        return False

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    if not (np.isfinite(mtx).all() and np.isfinite(dist).all() and np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return False

    tz = float(tvec[2, 0])
    min_z = 0.02
    if tz <= min_z:
        return False

    # Clamp radius so the ring stays in front of the camera.
    R, _ = cv2.Rodrigues(rvec)
    s = float(np.hypot(R[2, 0], R[2, 1]))
    if s > 1e-6:
        max_r = 0.95 * (tz - min_z) / s
        if max_r <= 0:
            return False
        radius_m = min(radius_m, max_r)

    if radius_m <= 0:
        return False

    pts_count = pick_project_circle_pts(mtx, tvec, radius_m, circle_pts, min_circle_pts)
    angles = np.linspace(0, 2 * np.pi, pts_count, endpoint=False)
    obj_pts = np.stack(
        [radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros_like(angles)],
        axis=1
    ).astype(np.float32)

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
    pts2 = img_pts.reshape(-1, 2)

    if not np.isfinite(pts2).all():
        return False

    h, w = frame.shape[:2]

    # Reject unstable projections that can create long screen-spanning streaks.
    max_abs_coord = float(np.max(np.abs(pts2)))
    if max_abs_coord > 1e6:
        return False

    min_xy = np.min(pts2, axis=0)
    max_xy = np.max(pts2, axis=0)
    box_w = float(max_xy[0] - min_xy[0])
    box_h = float(max_xy[1] - min_xy[1])

    if box_w > 2.5 * w or box_h > 2.5 * h:
        return False

    margin_x = 2.0 * w
    margin_y = 2.0 * h
    if (
        max_xy[0] < -margin_x or min_xy[0] > (w + margin_x) or
        max_xy[1] < -margin_y or min_xy[1] > (h + margin_y)
    ):
        return False

    seg = np.roll(pts2, -1, axis=0) - pts2
    seg_len = np.linalg.norm(seg, axis=1)
    if not np.isfinite(seg_len).all():
        return False
    med_seg = float(np.median(seg_len))
    max_seg = float(np.max(seg_len))
    frame_diag = float(np.hypot(w, h))
    if med_seg > 0 and max_seg > (10.0 * med_seg) and max_seg > (0.35 * frame_diag):
        return False

    poly = np.rint(pts2).astype(np.int32)

    if fill_alpha > 0.0:
        alpha_fill_polygon(frame, poly, color, fill_alpha)
    if thickness > 0 and alpha > 0.0:
        alpha_draw_polyline(frame, poly, color, thickness, alpha)
    return True


def draw_hit_pulse_2d(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    dt_ms: int,
    lead_ms: int,
    hold_ms: int,
    pulse_max_scale: float,
    hit_flash_ms: int,
) -> None:
    if dt_ms > lead_ms or dt_ms < -hold_ms:
        return

    cx, cy = x + w // 2, y + h // 2
    base_r = int(0.55 * max(w, h))

    if dt_ms >= 0:
        t = clamp01(dt_ms / float(lead_ms))
        scale = 1.0 + (pulse_max_scale - 1.0) * t
        alpha = 0.06 + 0.24 * (1.0 - t)
    else:
        t = clamp01((-dt_ms) / float(hold_ms))
        scale = 1.0 - 0.25 * t
        alpha = 0.22 * (1.0 - t)

    r = int(base_r * scale)

    alpha_draw_circle(frame, cx, cy, r, (0, 0, 255), 4, alpha)

    if abs(dt_ms) <= hit_flash_ms:
        alpha_draw_circle(frame, cx, cy, int(base_r * 1.05), (0, 0, 255), -1, 0.18)


def draw_hit_pulse_3d(
    frame: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dt_ms: int,
    lead_ms: int,
    hold_ms: int,
    pulse_max_scale: float,
    hit_flash_ms: int,
    hit_fill_alpha: float,
    drum_highlight_r_m: float,
    circle_pts: int,
    min_circle_pts: int,
    reference_r_m: float,
) -> None:
    if dt_ms > lead_ms or dt_ms < -hold_ms:
        return

    base_r = drum_highlight_r_m
    size_ratio = float(np.clip(base_r / max(reference_r_m, 1e-4), 0.5, 2.5))

    if dt_ms >= 0:
        t = smoothstep01(dt_ms / float(lead_ms))
        r = base_r * (1.0 + (pulse_max_scale - 1.0) * t)
        alpha = 0.30 + 0.20 * (1.0 - t)
        thickness = max(2, int(round((3 + 5 * (1.0 - t)) * size_ratio)))
    else:
        t = smoothstep01((-dt_ms) / float(hold_ms))
        r = base_r * (1.0 - 0.20 * t)
        alpha = 0.40 * (1.0 - t)
        thickness = max(2, int(round(6 * size_ratio)))

    flash = 0.0
    if abs(dt_ms) <= hit_flash_ms:
        flash = smoothstep01(1.0 - (abs(dt_ms) / float(max(1, hit_flash_ms))))

    ok = draw_projected_ring_safe(
        frame, mtx, dist, rvec, tvec,
        r, circle_pts, min_circle_pts,
        color=(0, 0, 255),
        alpha=min(0.95, alpha + 0.20 * flash),
        thickness=thickness,
        fill_alpha=hit_fill_alpha * flash,
    )
    if not ok:
        return


def draw_roll_2d(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    now_ms: int,
    pulse: Pulse,
    cfg: Config,
) -> None:
    # Clean roll: steady double ring so rapid repeats look intentional.
    # pulse.count roughly tracks how dense the roll is.
    strength = min(0.65, 0.42 + 0.05 * max(0, pulse.count - 2))
    outer_th = 12
    inner_th = 4

    # Outer ring
    draw_2d_ring(
        frame, x, y, w, h,
        color=(0, 0, 255),
        alpha=strength,
        thickness=outer_th,
        radius_scale=1.02,
    )

    # Inner ring for a cleaner "roll" look
    inner_alpha = min(0.45, 0.26 + 0.03 * max(0, pulse.count - 2))
    draw_2d_ring(
        frame, x, y, w, h,
        color=(0, 0, 255),
        alpha=inner_alpha,
        thickness=inner_th,
        radius_scale=0.90,
    )

    # Subtle flash when a new hit gets merged into the roll
    if pulse.last_hit_ms is not None and abs(now_ms - pulse.last_hit_ms) <= cfg.hit_flash_ms:
        draw_2d_ring(
            frame, x, y, w, h,
            color=(0, 0, 255),
            alpha=0.22,
            thickness=outer_th + 4,
            radius_scale=1.04,
        )


def draw_roll_3d(
    frame: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    now_ms: int,
    pulse: Pulse,
    drum_highlight_r_m: float,
    cfg: Config,
) -> None:
    # Clean roll: steady double ring in 3D.
    strength = min(0.65, 0.42 + 0.05 * max(0, pulse.count - 2))
    size_ratio = float(np.clip(drum_highlight_r_m / max(cfg.drum_highlight_r_m, 1e-4), 0.5, 2.5))
    outer_th = max(4, int(round(12 * size_ratio)))
    inner_th = max(2, int(round(4 * size_ratio)))

    r_outer = drum_highlight_r_m * 1.02
    r_inner = drum_highlight_r_m * 0.90

    ok = draw_projected_ring_safe(
        frame, mtx, dist, rvec, tvec,
        r_outer,
        cfg.circle_pts,
        cfg.min_circle_pts,
        color=(0, 0, 255),
        alpha=strength,
        thickness=outer_th
    )
    if not ok:
        return

    draw_projected_ring_safe(
        frame, mtx, dist, rvec, tvec,
        r_inner,
        cfg.circle_pts,
        cfg.min_circle_pts,
        color=(0, 0, 255),
        alpha=min(0.45, 0.26 + 0.03 * max(0, pulse.count - 2)),
        thickness=inner_th
    )

    if pulse.last_hit_ms is not None and abs(now_ms - pulse.last_hit_ms) <= cfg.hit_flash_ms:
        draw_projected_ring_safe(
            frame, mtx, dist, rvec, tvec,
            drum_highlight_r_m * 1.04,
            cfg.circle_pts,
            cfg.min_circle_pts,
            color=(0, 0, 255),
            alpha=0.22,
            thickness=max(4, outer_th + max(2, int(round(4 * size_ratio))))
        )


def make_aruco_detector(cfg: Config):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    if not cfg.corner_refine and hasattr(aruco, "CORNER_REFINE_NONE"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = True
    return aruco.ArucoDetector(dictionary, params)


def detect_markers(detector, gray: np.ndarray):
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}, None, corners
    detected = {int(mid): c for c, mid in zip(corners, ids.flatten())}
    return detected, ids.flatten().astype(int), corners


def detect_markers_fast(detector, gray: np.ndarray, detect_downscale: float):
    scale = float(detect_downscale)
    if scale >= 0.999:
        return detect_markers(detector, gray)

    small = cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detected_small, ids, corners_small = detect_markers(detector, small)
    inv = 1.0 / scale
    detected = {mid: (np.asarray(c, dtype=np.float32) * inv) for mid, c in detected_small.items()}
    corners = [(np.asarray(c, dtype=np.float32) * inv) for c in corners_small]
    return detected, ids, corners


def estimate_poses(corners, ids, mtx, dist, marker_length_m: float):
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if ids is None or mtx is None or dist is None:
        return poses

    aruco = getattr(cv2, "aruco", None)
    if aruco is not None and hasattr(aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_m, mtx, dist)
        for i, mid in enumerate(ids.flatten()):
            poses[int(mid)] = (rvecs[i], tvecs[i])
        return poses

    # OpenCV builds that removed estimatePoseSingleMarkers:
    # estimate each marker pose with solvePnP on the 4 marker corners.
    half = marker_length_m * 0.5
    obj_pts = np.array(
        [
            [-half, half, 0.0],   # top-left
            [half, half, 0.0],    # top-right
            [half, -half, 0.0],   # bottom-right
            [-half, -half, 0.0],  # bottom-left
        ],
        dtype=np.float32,
    )

    primary_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    fallback_flag = cv2.SOLVEPNP_ITERATIVE

    for i, mid in enumerate(ids.flatten()):
        img_pts = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
        if img_pts.shape != (4, 2):
            continue

        ok = False
        rvec = tvec = None
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist, flags=primary_flag)
        except Exception:
            ok = False

        if (not ok) and (primary_flag != fallback_flag):
            try:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist, flags=fallback_flag)
            except Exception:
                ok = False

        if not ok or rvec is None or tvec is None:
            continue
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            continue

        poses[int(mid)] = (rvec, tvec)

    return poses


def draw_fps(frame: np.ndarray, fps: float) -> None:
    text = f"{fps:4.1f} FPS"
    _h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = max(8, w - tw - 12)
    y = max(22, th + 10)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# PULSE QUEUE LOGIC
# ============================================================

def add_pulse(pulses_by_mid: Dict[int, Deque[Pulse]], mid: int, t_hit_ms: int, cfg: Config) -> None:
    q = pulses_by_mid.get(mid)
    if q is None:
        q = deque()
        pulses_by_mid[mid] = q

    if not q:
        q.append(Pulse(kind="hit", t_hit_ms=t_hit_ms))
        return

    last = q[-1]
    if last.kind == "roll":
        assert last.t_end_ms is not None
        if t_hit_ms - last.t_end_ms <= cfg.roll_merge_ms:
            last.t_end_ms = t_hit_ms
            last.count += 1
            last.last_hit_ms = t_hit_ms
        else:
            q.append(Pulse(kind="hit", t_hit_ms=t_hit_ms))
    else:
        if t_hit_ms - last.t_hit_ms <= cfg.roll_merge_ms:
            # Convert last + new into a roll
            q[-1] = Pulse(kind="roll", t_hit_ms=last.t_hit_ms, t_end_ms=t_hit_ms, count=2, last_hit_ms=t_hit_ms)
        else:
            q.append(Pulse(kind="hit", t_hit_ms=t_hit_ms))

    while len(q) > cfg.max_pulses_per_marker:
        q.popleft()


def cull_expired_pulses(pulses_by_mid: Dict[int, Deque[Pulse]], now_ms: int, cfg: Config) -> None:
    cutoff = now_ms - cfg.hold_ms - cfg.cleanup_margin_ms
    empty_keys: List[int] = []
    for mid, q in pulses_by_mid.items():
        while q:
            p = q[0]
            if p.kind == "hit":
                if p.t_hit_ms < cutoff:
                    q.popleft()
                    continue
                break
            else:
                assert p.t_end_ms is not None
                if p.t_end_ms < cutoff:
                    q.popleft()
                    continue
                break
        if not q:
            empty_keys.append(mid)

    for mid in empty_keys:
        pulses_by_mid.pop(mid, None)


def find_active_roll(q: Deque[Pulse], now_ms: int, cfg: Config) -> Optional[Pulse]:
    # Return the newest active roll, if any
    active: Optional[Pulse] = None
    for p in q:
        if p.kind != "roll":
            continue
        assert p.t_end_ms is not None
        if (now_ms >= p.t_hit_ms - cfg.lead_ms) and (now_ms <= p.t_end_ms + cfg.hold_ms):
            active = p
    return active


def pick_hit_pulses_to_render(q: Deque[Pulse], now_ms: int, cfg: Config) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (dt_future, dt_past) for hit pulses only:
      - dt_future: smallest dt >= 0
      - dt_past: largest dt < 0
    """
    dt_future: Optional[int] = None
    dt_past: Optional[int] = None
    for p in q:
        if p.kind != "hit":
            continue
        dt = p.t_hit_ms - now_ms
        if dt > cfg.lead_ms or dt < -cfg.hold_ms:
            continue
        if dt >= 0:
            if dt_future is None or dt < dt_future:
                dt_future = dt
        else:
            if dt_past is None or dt > dt_past:
                dt_past = dt
    return dt_future, dt_past


def marker_is_expected_soon(q: Deque[Pulse], now_ms: int, cfg: Config) -> bool:
    roll = find_active_roll(q, now_ms, cfg)
    if roll is not None:
        return True
    dt_future, dt_past = pick_hit_pulses_to_render(q, now_ms, cfg)
    return (dt_future is not None) or (dt_past is not None)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    args = parse_args()
    cfg = make_config(args)
    marker_id_to_name = {mid: name for name, mid in cfg.name_to_marker_id.items()}
    marker_id_to_radius_m = make_marker_radius_map(cfg)

    title, events = load_song(cfg.song_path)
    print(f"Loaded song: {title}, events: {len(events)}")
    print("Controls: s = start, r = reset, q = quit")
    if args.pi:
        print(
            "Pi preset: "
            f"{cfg.frame_w}x{cfg.frame_h}, detect_scale={cfg.detect_downscale:.2f}, "
            f"detect_every={cfg.detect_every_n_frames}, use_3d={cfg.use_3d}"
        )

    cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)

    detector = make_aruco_detector(cfg)

    mtx = dist = None
    use_3d = bool(cfg.use_3d)
    if use_3d:
        mtx, dist = safe_load_calibration(cfg.calib_path)
        if mtx is None or dist is None:
            raise RuntimeError(
                f"3D mode requires calibration at '{cfg.calib_path}'. "
                "Run calibrate_camera.py first."
            )

    running = False
    start_t: Optional[float] = None

    # Separate indices:
    # - ingest_idx: pushes events into pulse queues up through (now + lead + lookahead)
    # - next_idx: drives the banner text (next chord)
    ingest_idx = 0
    next_idx = 0

    pulses_by_mid: Dict[int, Deque[Pulse]] = {}
    frame_idx = 0

    # Cache marker detection/poses for skipped frames.
    cached_detected: Dict[int, np.ndarray] = {}
    cached_ids: Optional[np.ndarray] = None
    cached_corners: List[np.ndarray] = []
    cached_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    smoothed_pose_by_mid: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    cached_age = cfg.max_cached_detection_frames + 1

    fps_last_t = time.perf_counter()
    fps_frames = 0
    fps_value = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        run_detection = (
            cfg.detect_every_n_frames <= 1
            or (frame_idx % cfg.detect_every_n_frames) == 0
            or cached_age > cfg.max_cached_detection_frames
        )

        if run_detection:
            detected, ids, corners = detect_markers_fast(detector, gray, cfg.detect_downscale)
            cached_detected = detected
            cached_ids = ids
            cached_corners = corners
            cached_age = 0

            if use_3d and ids is not None:
                raw_poses = estimate_poses(corners, ids, mtx, dist, cfg.marker_length_m)
                next_smoothed: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                for mid, pose_now in raw_poses.items():
                    prev_pose = smoothed_pose_by_mid.get(mid)
                    if prev_pose is None:
                        next_smoothed[mid] = pose_now
                    else:
                        next_smoothed[mid] = smooth_pose(prev_pose, pose_now, cfg.pose_smooth_alpha)
                smoothed_pose_by_mid = next_smoothed
                cached_poses = next_smoothed
            else:
                smoothed_pose_by_mid = {}
                cached_poses = {}
        else:
            cached_age += 1

        if cached_age <= cfg.max_cached_detection_frames:
            detected = cached_detected
            ids = cached_ids
            corners = cached_corners
            poses = cached_poses if use_3d else {}
        else:
            detected = {}
            ids = None
            corners = []
            poses = {}

        now_ms = 0
        if running and start_t is not None:
            now_ms = int((time.perf_counter() - start_t) * 1000)

        # ------------------------------------------------------------
        # 1) Active time window ingestion (plus lookahead)
        # ------------------------------------------------------------
        if running:
            ingest_until = now_ms + cfg.lead_ms + cfg.lookahead_ms
            while ingest_idx < len(events) and events[ingest_idx]["t_ms"] <= ingest_until:
                e = events[ingest_idx]
                t_hit = int(e["t_ms"])
                drum_id = int(e["drum"])
                name = cfg.drum_id_to_name.get(drum_id)
                if name is not None:
                    mid = cfg.name_to_marker_id.get(name)
                    if mid is not None:
                        add_pulse(pulses_by_mid, mid, t_hit, cfg)
                ingest_idx += 1

            cull_expired_pulses(pulses_by_mid, now_ms, cfg)

        # ------------------------------------------------------------
        # Banner text (still based on "next chord")
        # ------------------------------------------------------------
        cue_text = f"{title} | Press 's' to start"
        cue_ok = True
        next_group: List[dict] = []
        next_t: Optional[int] = None
        dt_next: Optional[int] = None

        if running:
            next_idx = advance_idx_past_old(events, next_idx, now_ms, cfg.hold_ms)
            if next_idx >= len(events):
                cue_text = f"{title} | DONE"
            else:
                next_t, next_group = group_chord(events, next_idx, cfg.chord_eps_ms)
                dt_next = next_t - now_ms
                next_names = [cfg.drum_id_to_name.get(int(e["drum"]), f"DRUM {e['drum']}") for e in next_group]
                cue_text = f"{title} | NEXT: {' + '.join(next_names)} | in {max(dt_next, 0)}ms"

        # ------------------------------------------------------------
        # Draw detections: base green ring + pulses from per-marker queues
        # ------------------------------------------------------------
        for mid, c in detected.items():
            x, y, w, h = bbox_from_corners(c)
            roll: Optional[Pulse] = None
            drum_r_m = marker_id_to_radius_m.get(mid, cfg.drum_highlight_r_m)
            pose = poses.get(mid)
            has_pose = pose is not None

            # 3D-only: draw ring/animations only when a valid pose is available.
            if has_pose:
                rvec, tvec = pose
                draw_projected_ring_safe(
                    frame, mtx, dist, rvec, tvec,
                    drum_r_m,
                    cfg.circle_pts,
                    cfg.min_circle_pts,
                    color=(0, 255, 0),
                    alpha=cfg.base_ring_alpha,
                    thickness=cfg.base_ring_thickness,
                )

            # Render pulses for this marker, if any
            q = pulses_by_mid.get(mid)
            if running and q and has_pose:
                roll = find_active_roll(q, now_ms, cfg)
                if roll is not None:
                    rvec, tvec = pose
                    draw_roll_3d(frame, mtx, dist, rvec, tvec, now_ms, roll, drum_r_m, cfg)
                else:
                    dt_future, dt_past = pick_hit_pulses_to_render(q, now_ms, cfg)
                    # Draw up to two pulses: the nearest future and nearest past hit
                    rvec, tvec = pose
                    if dt_past is not None:
                        draw_hit_pulse_3d(
                            frame, mtx, dist, rvec, tvec, dt_past,
                            cfg.lead_ms, cfg.hold_ms, cfg.pulse_max_scale, cfg.hit_flash_ms, cfg.hit_fill_alpha,
                            drum_r_m, cfg.circle_pts, cfg.min_circle_pts, cfg.drum_highlight_r_m
                        )
                    if dt_future is not None:
                        draw_hit_pulse_3d(
                            frame, mtx, dist, rvec, tvec, dt_future,
                            cfg.lead_ms, cfg.hold_ms, cfg.pulse_max_scale, cfg.hit_flash_ms, cfg.hit_fill_alpha,
                            drum_r_m, cfg.circle_pts, cfg.min_circle_pts, cfg.drum_highlight_r_m
                        )
            # Label (show ROLL when a roll is active on this marker)
            label = marker_id_to_name.get(mid, f"ID {mid}")
            label_y = max(30, y - 8)

            if running and roll is not None:
                label_color = (0, 0, 255)
            elif has_pose:
                label_color = (0, 255, 0)
            else:
                label_color = (0, 255, 255)

            cv2.putText(
                frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2, cv2.LINE_AA
            )

            if running and roll is not None:
                cv2.putText(
                    frame, "ROLL", (x, label_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
                )
            elif not has_pose:
                cv2.putText(
                    frame, "NO POSE", (x, label_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
                )

        # ------------------------------------------------------------
        # Banner warning if expected marker isn't visible
        # Uses pulse queues rather than only the next chord
        # ------------------------------------------------------------
        if running and pulses_by_mid:
            missing_names: List[str] = []
            for mid, q in pulses_by_mid.items():
                if mid in poses:
                    continue
                if marker_is_expected_soon(q, now_ms, cfg):
                    missing_names.append(marker_id_to_name.get(mid, f"ID {mid}"))
            if missing_names:
                cue_ok = False
                cue_text = cue_text + f" | NO 3D POSE: {', '.join(sorted(set(missing_names)))}"

        draw_banner(frame, cue_text, ok=cue_ok)
        if cfg.show_fps:
            fps_frames += 1
            now_t = time.perf_counter()
            elapsed = now_t - fps_last_t
            if elapsed >= 0.5:
                fps_value = fps_frames / elapsed
                fps_frames = 0
                fps_last_t = now_t
            draw_fps(frame, fps_value)

        cv2.imshow("drum_overlay", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            running = True
            start_t = time.perf_counter()
            ingest_idx = 0
            next_idx = 0
            pulses_by_mid.clear()
            print("Song started")
        if key == ord("r"):
            running = False
            start_t = None
            ingest_idx = 0
            next_idx = 0
            pulses_by_mid.clear()
            print("Reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
