import argparse
import bisect
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Set, Tuple

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
    playback_speed: float = 1.0
    mode: str = "play"  # 'play', 'train', or 'score'

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

    # Stick tracking (optional camera-only hit detection)
    enable_stick_tracking: bool = False
    stick_hsv_lower: Tuple[int, int, int] = (35, 80, 80)
    stick_hsv_upper: Tuple[int, int, int] = (95, 255, 255)
    stick_min_area: int = 80
    stick_max_jump_px: int = 80
    stick_smooth_alpha: float = 0.5
    stick_detect_downscale: float = 0.7
    stick_debug: bool = False

    # Camera-only hit detection tuning
    hit_z_thresh_m: float = 0.015
    approach_speed_thresh_mps: float = 0.35
    cooldown_ms: int = 100
    rearm_z_m: float = 0.04

    # Visual feedback (correct/incorrect/hit flash)
    feedback_flash_ms: int = 120

    # Startup behavior
    auto_start_delay_s: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco drum overlay")
    parser.add_argument("--pi", action="store_true", help="Enable Raspberry Pi performance preset.")
    parser.add_argument("--mode", choices=["play", "train", "score"], default="play", help="Playback mode.")
    parser.add_argument("--train", action="store_true", help="Alias for --mode train.")
    parser.add_argument("--detect-scale", type=float, default=None, help="Marker detection scale in (0, 1].")
    parser.add_argument("--detect-every", type=int, default=None, help="Run marker detection every N frames (>= 1).")
    parser.add_argument("--speed", type=float, default=None, help="Song playback speed multiplier (e.g. 0.5, 1.0, 1.5, 2.0).")
    parser.add_argument("--stick-track", action="store_true", help="Enable stick-tip tracking and camera-only hit detection.")
    parser.add_argument("--stick-hsv-lower", type=str, default=None, help="Lower HSV bound as H,S,V.")
    parser.add_argument("--stick-hsv-upper", type=str, default=None, help="Upper HSV bound as H,S,V.")
    parser.add_argument("--stick-min-area", type=int, default=None, help="Minimum contour area for stick tip blob.")
    parser.add_argument("--stick-max-jump", type=int, default=None, help="Maximum per-frame stick tip jump in pixels.")
    parser.add_argument("--stick-debug", action="store_true", help="Draw stick debug overlays.")
    parser.add_argument(
        "--auto-start-delay",
        type=float,
        default=None,
        help="Auto-start after this many seconds (e.g. 3.0).",
    )
    parser.add_argument("--show-fps", action="store_true", help="Show realtime FPS in the overlay.")
    return parser.parse_args()


def parse_hsv_csv(text: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in text.split(",")]
        if len(parts) != 3:
            return fallback
        h = int(np.clip(parts[0], 0, 179))
        s = int(np.clip(parts[1], 0, 255))
        v = int(np.clip(parts[2], 0, 255))
        return (h, s, v)
    except Exception:
        return fallback


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
            "HIHAT": 3,
            "CRASH": 4,
        },
        drum_name_to_highlight_r_m={
            "SNARE": 0.14,
            "BASS": 0.20,
            "HIHAT": 0.17,
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
            stick_detect_downscale=0.5,
        )

    mode = "train" if bool(args.train) else str(args.mode)
    cfg = replace(cfg, mode=mode)

    if args.detect_scale is not None:
        cfg = replace(cfg, detect_downscale=max(0.25, min(1.0, float(args.detect_scale))))
    if args.detect_every is not None:
        cfg = replace(cfg, detect_every_n_frames=max(1, int(args.detect_every)))
    if args.speed is not None:
        cfg = replace(cfg, playback_speed=max(0.05, float(args.speed)))
    if args.stick_track:
        cfg = replace(cfg, enable_stick_tracking=True)
    if args.stick_hsv_lower is not None:
        cfg = replace(cfg, stick_hsv_lower=parse_hsv_csv(args.stick_hsv_lower, cfg.stick_hsv_lower))
    if args.stick_hsv_upper is not None:
        cfg = replace(cfg, stick_hsv_upper=parse_hsv_csv(args.stick_hsv_upper, cfg.stick_hsv_upper))
    if args.stick_min_area is not None:
        cfg = replace(cfg, stick_min_area=max(1, int(args.stick_min_area)))
    if args.stick_max_jump is not None:
        cfg = replace(cfg, stick_max_jump_px=max(1, int(args.stick_max_jump)))
    if args.stick_debug:
        cfg = replace(cfg, stick_debug=True)
    if args.show_fps:
        cfg = replace(cfg, show_fps=True)
    if args.auto_start_delay is not None:
        cfg = replace(cfg, auto_start_delay_s=max(0.0, float(args.auto_start_delay)))

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


@dataclass
class StickHitState:
    last_hit_ms: int = -1_000_000_000
    armed: bool = True
    prev_z_local: Optional[float] = None
    prev_radial_local: Optional[float] = None
    prev_t_ms: Optional[int] = None


@dataclass
class TrainState:
    idx: int = 0
    group_end_idx: int = 0
    target_t_ms: int = 0
    expected_mids: Set[int] = field(default_factory=set)
    remaining_mids: Set[int] = field(default_factory=set)
    chord_starts: List[int] = field(default_factory=list)
    chord_number: int = 0
    total_chords: int = 0
    correct_hits: int = 0
    mistakes: int = 0
    done: bool = False


@dataclass
class ScoreHit:
    hit_ms: int
    mid: int
    drum_name: str
    used: bool = False


@dataclass
class ScoreExpected:
    t_ms: int
    mid: int
    drum_name: str


@dataclass
class ScoreState:
    expected: List[ScoreExpected] = field(default_factory=list)
    hits: List[ScoreHit] = field(default_factory=list)
    next_expected_idx: int = 0
    offset_ms: float = 0.0
    score_sum: float = 0.0
    total_scored: int = 0
    perfect_count: int = 0
    good_count: int = 0
    ok_count: int = 0
    miss_count: int = 0
    skipped_count: int = 0
    extra_hits: int = 0
    last_judgment_text: str = ""
    last_judgment_until_ms: int = 0
    recent_scores: Deque[float] = field(default_factory=deque)
    milestone_text: str = ""
    milestone_color: Tuple[int, int, int] = (255, 255, 255)
    milestone_until_ms: int = 0
    last_milestone_total: int = 0
    finished: bool = False


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


class LatestFrameCapture:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture-thread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_seq
            return self._latest_frame.copy(), self._latest_seq

    def _run(self) -> None:
        fail_delay_s = 0.005
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(fail_delay_s)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_seq += 1


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


def build_chord_starts(events: List[dict], chord_eps_ms: int) -> List[int]:
    starts: List[int] = []
    i = 0
    while i < len(events):
        starts.append(i)
        _t, group = group_chord(events, i, chord_eps_ms)
        i += len(group)
    return starts


def events_group_to_mids(group: List[dict], cfg: Config) -> Set[int]:
    mids: Set[int] = set()
    for e in group:
        drum_id = int(e["drum"])
        name = cfg.drum_id_to_name.get(drum_id)
        if name is None:
            continue
        mid = cfg.name_to_marker_id.get(name)
        if mid is not None:
            mids.add(int(mid))
    return mids


def set_train_target(train: TrainState, events: List[dict], cfg: Config) -> None:
    while True:
        if train.idx >= len(events):
            train.done = True
            train.expected_mids.clear()
            train.remaining_mids.clear()
            train.group_end_idx = len(events)
            train.target_t_ms = 0
            train.chord_number = train.total_chords
            return

        t_ms, group = group_chord(events, train.idx, cfg.chord_eps_ms)
        train.target_t_ms = int(t_ms)
        train.group_end_idx = train.idx + len(group)
        mids = events_group_to_mids(group, cfg)
        train.expected_mids = set(mids)
        train.remaining_mids = set(mids)
        train.chord_number = bisect.bisect_right(train.chord_starts, train.idx)

        # Skip unmapped chords so train mode never gets stuck.
        if train.remaining_mids:
            train.done = False
            return
        train.idx = train.group_end_idx


def skip_train_target(train: TrainState, events: List[dict], cfg: Config) -> None:
    if train.done:
        return
    if train.group_end_idx > train.idx:
        train.idx = train.group_end_idx
    else:
        train.idx += 1
    set_train_target(train, events, cfg)


def set_feedback(
    feedback_by_mid: Dict[int, Tuple[Tuple[int, int, int], int]],
    mid: int,
    color: Tuple[int, int, int],
    now_ms: int,
    flash_ms: int,
) -> None:
    feedback_by_mid[int(mid)] = (color, int(now_ms + flash_ms))


def cleanup_feedback(
    feedback_by_mid: Dict[int, Tuple[Tuple[int, int, int], int]],
    now_ms: int,
) -> None:
    stale = [mid for mid, (_c, t_exp) in feedback_by_mid.items() if now_ms > t_exp]
    for mid in stale:
        feedback_by_mid.pop(mid, None)


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


def draw_center_popup(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    if not text:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.25
    thick = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad_x = 26
    pad_y = 22
    box_w = tw + (2 * pad_x)
    box_h = th + baseline + (2 * pad_y)
    x0 = max(0, (w - box_w) // 2)
    y0 = max(0, (h - box_h) // 2)
    x1 = min(w, x0 + box_w)
    y1 = min(h, y0 + box_h)

    alpha_fill_rect(frame, x0, y0, x1, y1, (0, 0, 0), 0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    tx = x0 + (box_w - tw) // 2
    ty = y0 + pad_y + th
    cv2.putText(frame, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


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
    # Improve long-distance/small-marker detect stability.
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = 0.015
    if hasattr(params, "maxMarkerPerimeterRate"):
        params.maxMarkerPerimeterRate = 4.0
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 53
    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 4
    if hasattr(params, "minDistanceToBorder"):
        params.minDistanceToBorder = 1
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True
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


def detect_stick_tip(
    frame_bgr: np.ndarray,
    prev_tip_full: Optional[np.ndarray],
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    scale = float(np.clip(cfg.stick_detect_downscale, 0.20, 1.0))
    if scale < 0.999:
        small = cv2.resize(frame_bgr, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lower = np.array(cfg.stick_hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.stick_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_small = float(cfg.stick_min_area) * (scale * scale)
    candidates: List[Tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area_small:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0.0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        candidates.append((area, np.array([cx, cy], dtype=np.float32)))

    if not candidates:
        return None, (mask if cfg.stick_debug else None)

    picked_small: Optional[np.ndarray] = None
    if prev_tip_full is None:
        picked_small = max(candidates, key=lambda x: x[0])[1]
    else:
        prev_small = np.asarray(prev_tip_full, dtype=np.float32).reshape(2) * scale
        candidates_with_dist = [
            (float(np.linalg.norm(cxy - prev_small)), area, cxy)
            for area, cxy in candidates
        ]
        candidates_with_dist.sort(key=lambda x: x[0])
        max_jump_small = float(cfg.stick_max_jump_px) * scale
        in_jump = [x for x in candidates_with_dist if x[0] <= max_jump_small]
        if in_jump:
            picked_small = in_jump[0][2]
        elif len(candidates_with_dist) == 1:
            # If this is the only candidate, allow large jump reacquisition.
            picked_small = candidates_with_dist[0][2]
        else:
            return None, (mask if cfg.stick_debug else None)

    if picked_small is None:
        return None, (mask if cfg.stick_debug else None)

    tip_full = picked_small / scale
    tip_full = np.asarray(tip_full, dtype=np.float32).reshape(2)

    if prev_tip_full is not None:
        prev = np.asarray(prev_tip_full, dtype=np.float32).reshape(2)
        a = float(clamp01(cfg.stick_smooth_alpha))
        tip_full = (1.0 - a) * prev + a * tip_full

    return tip_full, (mask if cfg.stick_debug else None)


def polygon_area(pts_4x2: np.ndarray) -> float:
    pts = np.asarray(pts_4x2, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def validate_tracked_corners(
    prev_pts_4x2: np.ndarray,
    new_pts_4x2: np.ndarray,
    status_4: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> bool:
    prev_pts = np.asarray(prev_pts_4x2, dtype=np.float32).reshape(4, 2)
    new_pts = np.asarray(new_pts_4x2, dtype=np.float32).reshape(4, 2)
    status = np.asarray(status_4).reshape(-1)

    if status.shape[0] != 4 or not np.all(status == 1):
        return False
    if not (np.isfinite(prev_pts).all() and np.isfinite(new_pts).all()):
        return False

    margin = 20.0
    if np.any(new_pts[:, 0] < -margin) or np.any(new_pts[:, 0] > (frame_w + margin)):
        return False
    if np.any(new_pts[:, 1] < -margin) or np.any(new_pts[:, 1] > (frame_h + margin)):
        return False

    area_prev = polygon_area(prev_pts)
    area_new = polygon_area(new_pts)
    if area_prev < 4.0 or area_new < 4.0:
        return False
    area_ratio = area_new / max(area_prev, 1e-6)
    if area_ratio < 0.45 or area_ratio > 2.20:
        return False

    prev_edges = np.roll(prev_pts, -1, axis=0) - prev_pts
    new_edges = np.roll(new_pts, -1, axis=0) - new_pts
    prev_len = np.linalg.norm(prev_edges, axis=1)
    new_len = np.linalg.norm(new_edges, axis=1)
    if np.any(prev_len < 1.2) or np.any(new_len < 1.2):
        return False
    if np.any(new_len > (prev_len * 2.5 + 8.0)):
        return False

    corner_jump = np.linalg.norm(new_pts - prev_pts, axis=1)
    max_jump = float(np.max(corner_jump))
    jump_cap = max(35.0, float(1.8 * np.median(prev_len)))
    if max_jump > jump_cap:
        return False

    return True


def build_detect_outputs_from_corner_map(
    corner_map: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
    if not corner_map:
        return {}, None, []

    ids_sorted = np.array(sorted(corner_map.keys()), dtype=np.int32)
    corners: List[np.ndarray] = []
    detected: Dict[int, np.ndarray] = {}
    for mid in ids_sorted.tolist():
        c1x4x2 = np.asarray(corner_map[mid], dtype=np.float32).reshape(1, 4, 2)
        corners.append(c1x4x2)
        detected[int(mid)] = c1x4x2
    return detected, ids_sorted, corners


def track_marker_corners_lk(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_corner_map: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    if prev_gray is None or curr_gray is None or not prev_corner_map:
        return {}

    mids = sorted(prev_corner_map.keys())
    prev_pts_all = np.concatenate(
        [np.asarray(prev_corner_map[mid], dtype=np.float32).reshape(4, 2) for mid in mids],
        axis=0
    ).reshape(-1, 1, 2)

    lk_win = (21, 21)
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    next_pts_all, status_all, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts_all,
        None,
        winSize=lk_win,
        maxLevel=3,
        criteria=lk_criteria,
    )
    if next_pts_all is None or status_all is None:
        return {}

    prev_pts_all = prev_pts_all.reshape(-1, 2)
    next_pts_all = next_pts_all.reshape(-1, 2)
    status_all = status_all.reshape(-1)

    h, w = curr_gray.shape[:2]
    out: Dict[int, np.ndarray] = {}
    for i, mid in enumerate(mids):
        j0 = i * 4
        j1 = j0 + 4
        prev_pts = prev_pts_all[j0:j1]
        next_pts = next_pts_all[j0:j1]
        status = status_all[j0:j1]
        if validate_tracked_corners(prev_pts, next_pts, status, w, h):
            out[int(mid)] = next_pts.astype(np.float32)
    return out


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


def ray_plane_intersect(
    ray_origin_cam: np.ndarray,
    ray_dir_cam: np.ndarray,
    plane_point_cam: np.ndarray,
    plane_normal_cam: np.ndarray,
    eps: float = 1e-8,
) -> Optional[np.ndarray]:
    o = np.asarray(ray_origin_cam, dtype=np.float64).reshape(3)
    d = np.asarray(ray_dir_cam, dtype=np.float64).reshape(3)
    p0 = np.asarray(plane_point_cam, dtype=np.float64).reshape(3)
    n = np.asarray(plane_normal_cam, dtype=np.float64).reshape(3)

    denom = float(np.dot(n, d))
    if abs(denom) < eps:
        return None
    t = float(np.dot(n, p0 - o) / denom)
    if t <= 0.0:
        return None
    return o + t * d


def update_hit_states(
    tip_uv: Optional[np.ndarray],
    mtx: Optional[np.ndarray],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    marker_id_to_radius_m: Dict[int, float],
    hit_state_by_mid: Dict[int, StickHitState],
    now_wall_ms: int,
    cfg: Config,
) -> Tuple[Optional[int], Optional[Tuple[int, float, float, float]]]:
    if tip_uv is None or mtx is None or not poses:
        return None, None

    u, v = np.asarray(tip_uv, dtype=np.float64).reshape(2)
    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    if fx <= 1e-8 or fy <= 1e-8:
        return None, None

    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_dir = np.array([x, y, 1.0], dtype=np.float64)
    ray_norm = float(np.linalg.norm(ray_dir))
    if ray_norm <= 1e-12:
        return None, None
    ray_dir /= ray_norm

    ray_origin = np.zeros(3, dtype=np.float64)
    candidates: List[Tuple[int, float, float]] = []  # (mid, score, radial)
    closest_info: Optional[Tuple[int, float, float, float]] = None

    for mid, pose in poses.items():
        rvec, tvec = pose
        r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if not (np.isfinite(r).all() and np.isfinite(t).all()):
            continue

        R, _ = cv2.Rodrigues(r)
        n_cam = R[:, 2]
        p0_cam = t.reshape(3)
        p_cam = ray_plane_intersect(ray_origin, ray_dir, p0_cam, n_cam)
        if p_cam is None:
            continue

        p_local = R.T @ (p_cam - p0_cam)
        x_local = float(p_local[0])
        y_local = float(p_local[1])
        z_local = float(p_local[2])
        radial = float(np.hypot(x_local, y_local))

        state = hit_state_by_mid.get(mid)
        if state is None:
            state = StickHitState()
            hit_state_by_mid[mid] = state

        prev_z = state.prev_z_local
        prev_radial = state.prev_radial_local
        prev_t = state.prev_t_ms
        vz = 0.0
        vr = 0.0
        if prev_z is not None and prev_t is not None and now_wall_ms > prev_t:
            dt_s = max((now_wall_ms - prev_t) / 1000.0, 1e-4)
            vz = (z_local - prev_z) / dt_s
        if prev_radial is not None and prev_t is not None and now_wall_ms > prev_t:
            dt_s = max((now_wall_ms - prev_t) / 1000.0, 1e-4)
            vr = (radial - prev_radial) / dt_s

        drum_r_m = float(marker_id_to_radius_m.get(mid, cfg.drum_highlight_r_m))
        rearm_margin = max(0.005, float(cfg.rearm_z_m))
        enter_margin = max(0.003, min(float(cfg.rearm_z_m), 0.35 * drum_r_m))
        generous_margin = max(0.006, 0.35 * enter_margin)
        hit_radius = drum_r_m + generous_margin
        z_tol = max(0.03, 2.0 * float(cfg.hit_z_thresh_m))
        z_ok = abs(z_local) <= z_tol
        if radial > (drum_r_m + rearm_margin):
            state.armed = True

        cooldown_ok = (now_wall_ms - state.last_hit_ms) >= int(cfg.cooldown_ms)
        crossing = (prev_radial is not None) and (prev_radial > (drum_r_m + enter_margin)) and (radial <= hit_radius)
        radial_delta = 0.0 if prev_radial is None else float(prev_radial - radial)
        speed_ok = (vr < -cfg.approach_speed_thresh_mps) or (radial_delta > max(0.004, 0.15 * enter_margin))
        moving_ok = radial_delta > -max(0.003, 0.20 * enter_margin)
        inside_ok = (prev_radial is not None) and (radial <= hit_radius)
        candidate_hit = (
            state.armed
            and inside_ok
            and (crossing or moving_ok or speed_ok)
            and z_ok
            and cooldown_ok
        )

        if candidate_hit:
            # Prefer the marker where the stick tip is deeper inside the target ring.
            inside_depth = max(0.0, hit_radius - radial)
            score = (40.0 * inside_depth) + (2.0 * max(0.0, radial_delta)) + (0.05 * max(0.0, -vr))
            candidates.append((int(mid), score, radial))

        if closest_info is None or radial < closest_info[2]:
            closest_info = (int(mid), z_local, radial, vr)

        state.prev_z_local = z_local
        state.prev_radial_local = radial
        state.prev_t_ms = int(now_wall_ms)

    if not candidates:
        return None, closest_info

    candidates.sort(key=lambda x: (-x[1], x[2]))
    hit_mid = int(candidates[0][0])
    st = hit_state_by_mid.get(hit_mid)
    if st is not None:
        st.last_hit_ms = int(now_wall_ms)
        st.armed = False
    return hit_mid, closest_info


def update_train_mode(
    train: TrainState,
    hit_mid: Optional[int],
    events: List[dict],
    cfg: Config,
) -> Tuple[Optional[int], Optional[int], bool]:
    if train.done:
        return None, None, False

    if train.group_end_idx <= train.idx and train.idx < len(events):
        set_train_target(train, events, cfg)
    if train.done:
        return None, None, False

    if hit_mid is None:
        return None, None, False

    if hit_mid in train.remaining_mids:
        train.remaining_mids.remove(hit_mid)
        train.correct_hits += 1
        if not train.remaining_mids:
            train.idx = train.group_end_idx
            set_train_target(train, events, cfg)
            return int(hit_mid), None, True
        return int(hit_mid), None, False

    train.mistakes += 1
    return None, int(hit_mid), False


def build_score_expected_events(events: List[dict], cfg: Config) -> List[ScoreExpected]:
    out: List[ScoreExpected] = []
    i = 0
    while i < len(events):
        t_chord, group = group_chord(events, i, cfg.chord_eps_ms)
        for e in group:
            drum_id = int(e["drum"])
            drum_name = cfg.drum_id_to_name.get(drum_id)
            if drum_name is None:
                continue
            mid = cfg.name_to_marker_id.get(drum_name)
            if mid is None:
                continue
            out.append(ScoreExpected(t_ms=int(t_chord), mid=int(mid), drum_name=drum_name))
        i += len(group)
    return out


def score_dt(abs_dt_ms: int) -> Tuple[float, str]:
    dt = abs(int(abs_dt_ms))
    if dt <= 60:
        return 1.00, "PERFECT"
    if dt <= 120:
        return 0.90, "GOOD"
    if dt <= 200:
        return 0.75, "OK"
    if dt <= 260:
        return 0.60, "OK"
    return 0.00, "MISS"


def update_offset(
    offset_ms: float,
    expected_ms: int,
    hit_ms: int,
    alpha: float = 0.12,
    clamp_ms: float = 350.0,
) -> float:
    a = clamp01(alpha)
    target = float(hit_ms - expected_ms)
    out = (1.0 - a) * float(offset_ms) + a * target
    return float(np.clip(out, -clamp_ms, clamp_ms))


def add_scored_hit(score: ScoreState, hit_score: float, now_ms: int, flash_ms: int = 1800) -> None:
    score.total_scored += 1
    score.score_sum += float(hit_score)
    score.recent_scores.append(float(hit_score))
    while len(score.recent_scores) > 10:
        score.recent_scores.popleft()

    if score.total_scored % 10 != 0:
        return
    if score.total_scored == score.last_milestone_total:
        return

    recent_avg = float(sum(score.recent_scores) / max(1, len(score.recent_scores)))
    recent_acc = 100.0 * recent_avg
    if recent_acc >= 90.0:
        verdict = "Excellent"
        color = (0, 255, 0)
    elif recent_acc >= 75.0:
        verdict = "Good"
        color = (0, 220, 120)
    elif recent_acc >= 60.0:
        verdict = "Okay"
        color = (0, 255, 255)
    else:
        verdict = "Bad"
        color = (0, 0, 255)

    score.milestone_text = f"Last 10 acc: {recent_acc:.0f}% ({verdict})"
    score.milestone_color = color
    score.milestone_until_ms = int(now_ms + flash_ms)
    score.last_milestone_total = score.total_scored


def match_expected_to_hits(
    expected_t_ms: int,
    expected_mid: int,
    hits: List[ScoreHit],
    offset_ms: float,
    early_ms: int = 220,
    late_ms: int = 260,
) -> Tuple[Optional[int], Optional[int], bool]:
    best_idx: Optional[int] = None
    best_dt: Optional[int] = None
    best_rank = (2, 1e9)

    for i, h in enumerate(hits):
        if h.used:
            continue
        corrected_hit_ms = float(h.hit_ms) - float(offset_ms)
        dt = corrected_hit_ms - float(expected_t_ms)
        if dt < -float(early_ms) or dt > float(late_ms):
            continue
        is_wrong = 0 if int(h.mid) == int(expected_mid) else 1
        rank = (is_wrong, abs(dt))
        if rank < best_rank:
            best_rank = rank
            best_idx = i
            best_dt = int(round(dt))

    is_correct = bool(best_idx is not None and best_rank[0] == 0)
    return best_idx, best_dt, is_correct


def had_pose_in_window(samples: Optional[Deque[int]], t_min_ms: int, t_max_ms: int) -> bool:
    if samples is None:
        return False
    for ts in samples:
        if ts < t_min_ms:
            continue
        if ts <= t_max_ms:
            return True
        break
    return False


def process_score_mode_events(
    score: ScoreState,
    now_ms: int,
    wall_ms: int,
    pose_seen_by_mid: Dict[int, Deque[int]],
    feedback_by_mid: Dict[int, Tuple[Tuple[int, int, int], int]],
    marker_id_to_name: Dict[int, str],
    cfg: Config,
) -> None:
    if score.finished:
        return

    early_ms = 220
    late_ms = 260
    judge_flash_ms = 500

    while score.next_expected_idx < len(score.expected):
        exp = score.expected[score.next_expected_idx]
        if now_ms < (exp.t_ms + late_ms):
            break

        win_lo = exp.t_ms - early_ms
        win_hi = exp.t_ms + late_ms
        if not had_pose_in_window(pose_seen_by_mid.get(exp.mid), win_lo, win_hi):
            score.skipped_count += 1
            score.last_judgment_text = f"{exp.drum_name} SKIP (no pose)"
            score.last_judgment_until_ms = int(now_ms + judge_flash_ms)
            score.next_expected_idx += 1
            continue

        idx, dt_ms, is_correct = match_expected_to_hits(
            exp.t_ms,
            exp.mid,
            score.hits,
            score.offset_ms,
            early_ms=early_ms,
            late_ms=late_ms,
        )
        if idx is None or dt_ms is None:
            add_scored_hit(score, 0.00, now_ms)
            score.miss_count += 1
            score.last_judgment_text = f"{exp.drum_name} MISS"
            score.last_judgment_until_ms = int(now_ms + judge_flash_ms)
            score.next_expected_idx += 1
            continue

        hit = score.hits[idx]
        hit.used = True

        if is_correct:
            hit_score, label = score_dt(abs(dt_ms))
            add_scored_hit(score, hit_score, now_ms)
            if label == "PERFECT":
                score.perfect_count += 1
            elif label == "GOOD":
                score.good_count += 1
            elif label == "OK":
                score.ok_count += 1
            else:
                score.miss_count += 1
            score.offset_ms = update_offset(score.offset_ms, exp.t_ms, hit.hit_ms)
            set_feedback(feedback_by_mid, exp.mid, (0, 255, 0), wall_ms, cfg.feedback_flash_ms)
            score.last_judgment_text = f"{exp.drum_name} {label} (dt={dt_ms:+d}ms)"
        else:
            wrong_score = 0.25 if abs(dt_ms) <= 120 else 0.00
            add_scored_hit(score, wrong_score, now_ms)
            score.miss_count += 1
            wrong_name = marker_id_to_name.get(hit.mid, hit.drum_name)
            set_feedback(feedback_by_mid, hit.mid, (0, 0, 255), wall_ms, cfg.feedback_flash_ms)
            score.last_judgment_text = f"{exp.drum_name} MISS ({wrong_name}, dt={dt_ms:+d}ms)"

        score.last_judgment_until_ms = int(now_ms + judge_flash_ms)
        score.next_expected_idx += 1

    if score.next_expected_idx >= len(score.expected):
        score.finished = True
        score.extra_hits = sum(1 for h in score.hits if not h.used)


def update_score_mode_ui(title: str, score: ScoreState, running: bool, now_ms: int) -> str:
    acc = 0.0
    if score.total_scored > 0:
        acc = 100.0 * (score.score_sum / float(score.total_scored))

    stats = (
        f"Acc {acc:5.1f}% | "
        f"P/G/O/M {score.perfect_count}/{score.good_count}/{score.ok_count}/{score.miss_count} | "
        f"offset {score.offset_ms:+.0f}ms | skipped {score.skipped_count}"
    )
    text = (
        f"{title} | MODE SCORE | {stats}"
    )
    if not running:
        text = f"{title} | MODE SCORE | Press 's' to start | {stats}"
    if running and score.last_judgment_text and now_ms <= score.last_judgment_until_ms:
        text += f" | {score.last_judgment_text}"
    if running and score.milestone_text and now_ms <= score.milestone_until_ms:
        text += f" | {score.milestone_text}"
    if score.finished:
        text += " | DONE"
    return text


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
    if cfg.mode == "train":
        print("Controls: s = start, n = next chord, r = reset, q = quit")
    else:
        print("Controls: s = start, r = reset, q = quit")
    print(f"Playback speed: {cfg.playback_speed:.2f}x")
    mode_blocked = (cfg.mode in ("train", "score")) and (not cfg.enable_stick_tracking)
    if (cfg.mode != "play") or cfg.enable_stick_tracking:
        print(f"Mode: {cfg.mode}")
        print(f"Stick tracking: {'ON' if cfg.enable_stick_tracking else 'OFF'}")
    if mode_blocked:
        print(f"{cfg.mode.upper()} MODE WARNING: enable --stick-track to start.")
    if args.pi:
        print(
            "Pi preset: "
            f"{cfg.frame_w}x{cfg.frame_h}, detect_scale={cfg.detect_downscale:.2f}, "
            f"detect_every={cfg.detect_every_n_frames}, use_3d={cfg.use_3d}, speed={cfg.playback_speed:.2f}x"
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

    capture = LatestFrameCapture(cap)
    capture.start()

    running = False
    start_t: Optional[float] = None

    # Separate indices:
    # - ingest_idx: pushes events into pulse queues up through (now + lead + lookahead)
    # - next_idx: drives the banner text (next chord)
    ingest_idx = 0
    next_idx = 0

    pulses_by_mid: Dict[int, Deque[Pulse]] = {}
    frame_idx = 0
    last_frame_seq = 0

    # Detection + tracking state.
    tracked_corner_map: Dict[int, np.ndarray] = {}
    prev_gray_for_track: Optional[np.ndarray] = None
    smoothed_pose_by_mid: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Optional stick-tip tracking + camera-only hit detection state.
    stick_tip_full: Optional[np.ndarray] = None
    stick_mask_debug: Optional[np.ndarray] = None
    closest_stick_info: Optional[Tuple[int, float, float, float]] = None
    hit_state_by_mid: Dict[int, StickHitState] = {}
    feedback_by_mid: Dict[int, Tuple[Tuple[int, int, int], int]] = {}

    # Train mode state.
    train_state: Optional[TrainState] = None
    if cfg.mode == "train":
        chord_starts = build_chord_starts(events, cfg.chord_eps_ms)
        train_state = TrainState(chord_starts=chord_starts, total_chords=len(chord_starts))
        set_train_target(train_state, events, cfg)

    # Score mode state.
    score_state: Optional[ScoreState] = None
    pose_seen_by_mid: Dict[int, Deque[int]] = {}
    score_summary_printed = False
    if cfg.mode == "score":
        expected = build_score_expected_events(events, cfg)
        score_state = ScoreState(expected=expected)

    auto_start_at_t: Optional[float] = None
    if cfg.auto_start_delay_s > 0.0:
        auto_start_at_t = time.perf_counter() + cfg.auto_start_delay_s
        print(f"Auto-start enabled: starting in {cfg.auto_start_delay_s:.1f}s")

    def reset_mode_state() -> None:
        nonlocal ingest_idx, next_idx, score_summary_printed
        ingest_idx = 0
        next_idx = 0
        pulses_by_mid.clear()
        hit_state_by_mid.clear()
        feedback_by_mid.clear()
        pose_seen_by_mid.clear()
        score_summary_printed = False

        if train_state is not None:
            train_state.idx = 0
            train_state.group_end_idx = 0
            train_state.target_t_ms = 0
            train_state.expected_mids.clear()
            train_state.remaining_mids.clear()
            train_state.correct_hits = 0
            train_state.mistakes = 0
            train_state.done = False
            set_train_target(train_state, events, cfg)

        if score_state is not None:
            score_state.hits.clear()
            score_state.next_expected_idx = 0
            score_state.offset_ms = 0.0
            score_state.score_sum = 0.0
            score_state.total_scored = 0
            score_state.perfect_count = 0
            score_state.good_count = 0
            score_state.ok_count = 0
            score_state.miss_count = 0
            score_state.skipped_count = 0
            score_state.extra_hits = 0
            score_state.last_judgment_text = ""
            score_state.last_judgment_until_ms = 0
            score_state.recent_scores.clear()
            score_state.milestone_text = ""
            score_state.milestone_color = (255, 255, 255)
            score_state.milestone_until_ms = 0
            score_state.last_milestone_total = 0
            score_state.finished = False

    def start_session(from_auto: bool = False) -> bool:
        nonlocal running, start_t, stick_tip_full, auto_start_at_t
        if mode_blocked:
            return False

        running = True
        start_t = time.perf_counter()
        stick_tip_full = None
        auto_start_at_t = None
        reset_mode_state()

        if cfg.mode == "train":
            print("Training started" + (" (auto)" if from_auto else ""))
        elif cfg.mode == "score":
            print("Score mode started" + (" (auto)" if from_auto else ""))
        else:
            print("Song started" + (" (auto)" if from_auto else ""))
        return True

    fps_last_t = time.perf_counter()
    fps_frames = 0
    fps_value = 0.0

    try:
        while True:
            frame, frame_seq = capture.get_latest()
            if frame is None or frame_seq == last_frame_seq:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(0.001)
                continue

            last_frame_seq = frame_seq
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            run_detection = (
                cfg.detect_every_n_frames <= 1
                or (frame_idx % cfg.detect_every_n_frames) == 0
                or prev_gray_for_track is None
                or not tracked_corner_map
            )

            if run_detection:
                detected_raw, _ids_raw, _corners_raw = detect_markers_fast(detector, gray, cfg.detect_downscale)
                next_corner_map: Dict[int, np.ndarray] = {}
                for mid, c in detected_raw.items():
                    c4 = np.asarray(c, dtype=np.float32).reshape(4, 2)
                    if np.isfinite(c4).all():
                        next_corner_map[int(mid)] = c4
                tracked_corner_map = next_corner_map
            else:
                tracked_corner_map = track_marker_corners_lk(
                    prev_gray_for_track,
                    gray,
                    tracked_corner_map,
                )

            detected, ids, corners = build_detect_outputs_from_corner_map(tracked_corner_map)
            prev_gray_for_track = gray

            poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            if use_3d and ids is not None:
                raw_poses = estimate_poses(corners, ids, mtx, dist, cfg.marker_length_m)
                next_smoothed: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                pose_alpha = cfg.pose_smooth_alpha
                if not run_detection:
                    pose_alpha = min(0.80, cfg.pose_smooth_alpha + 0.10)
                for mid, pose_now in raw_poses.items():
                    prev_pose = smoothed_pose_by_mid.get(mid)
                    if prev_pose is None:
                        next_smoothed[mid] = pose_now
                    else:
                        next_smoothed[mid] = smooth_pose(prev_pose, pose_now, pose_alpha)
                smoothed_pose_by_mid = next_smoothed
                poses = next_smoothed
            else:
                smoothed_pose_by_mid = {}

            loop_t = time.perf_counter()
            wall_ms = int(loop_t * 1000.0)

            if (not running) and (auto_start_at_t is not None) and (not mode_blocked):
                if loop_t >= auto_start_at_t:
                    start_session(from_auto=True)

            now_ms = 0
            if running and start_t is not None:
                elapsed_ms = (loop_t - start_t) * 1000.0
                if cfg.mode in ("play", "score"):
                    now_ms = int(elapsed_ms * cfg.playback_speed)
                else:
                    now_ms = int(elapsed_ms)

            hit_mid: Optional[int] = None
            closest_stick_info = None
            if cfg.enable_stick_tracking:
                tip_now, stick_mask_debug = detect_stick_tip(frame, stick_tip_full, cfg)
                stick_tip_full = tip_now
                hit_mid_raw, closest_stick_info = update_hit_states(
                    stick_tip_full,
                    mtx,
                    poses,
                    marker_id_to_radius_m,
                    hit_state_by_mid,
                    wall_ms,
                    cfg,
                )
                if running:
                    hit_mid = hit_mid_raw
                    if hit_mid is not None:
                        hit_name = marker_id_to_name.get(hit_mid, f"ID {hit_mid}")
                        print(f"HIT: {hit_name} at {now_ms}ms")
                        if cfg.mode == "score" and score_state is not None:
                            score_state.hits.append(ScoreHit(hit_ms=int(now_ms), mid=int(hit_mid), drum_name=hit_name))

            # ------------------------------------------------------------
            # 1) Active time window ingestion (plus lookahead)
            # ------------------------------------------------------------
            if running and cfg.mode in ("play", "score"):
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
            # 2) Train/Score progression from hit events
            # ------------------------------------------------------------
            if running and cfg.mode == "score":
                # Keep short pose-visibility history for skip logic.
                hist_cutoff = now_ms - 2200
                for mid in list(pose_seen_by_mid.keys()):
                    dq = pose_seen_by_mid[mid]
                    while dq and dq[0] < hist_cutoff:
                        dq.popleft()
                    if not dq and mid not in poses:
                        pose_seen_by_mid.pop(mid, None)
                for mid in poses.keys():
                    dq = pose_seen_by_mid.setdefault(int(mid), deque())
                    dq.append(int(now_ms))

            if running and cfg.mode == "train" and train_state is not None:
                correct_mid = wrong_mid = None
                if hit_mid is not None:
                    correct_mid, wrong_mid, _advanced = update_train_mode(train_state, hit_mid, events, cfg)
                if correct_mid is not None:
                    set_feedback(feedback_by_mid, correct_mid, (0, 255, 0), wall_ms, cfg.feedback_flash_ms)
                    print(f"TRAIN OK: {marker_id_to_name.get(correct_mid, f'ID {correct_mid}')}")
                if wrong_mid is not None:
                    set_feedback(feedback_by_mid, wrong_mid, (0, 0, 255), wall_ms, cfg.feedback_flash_ms)
                    print(f"TRAIN MISS: {marker_id_to_name.get(wrong_mid, f'ID {wrong_mid}')}")
            elif running and cfg.mode == "play" and hit_mid is not None:
                set_feedback(feedback_by_mid, hit_mid, (0, 255, 0), wall_ms, cfg.feedback_flash_ms)
            elif running and cfg.mode == "score" and score_state is not None:
                process_score_mode_events(
                    score_state,
                    now_ms,
                    wall_ms,
                    pose_seen_by_mid,
                    feedback_by_mid,
                    marker_id_to_name,
                    cfg,
                )
                if score_state.finished and not score_summary_printed:
                    acc = 0.0
                    if score_state.total_scored > 0:
                        acc = 100.0 * (score_state.score_sum / float(score_state.total_scored))
                    print("=== SCORE SUMMARY ===")
                    print(f"Accuracy: {acc:.2f}%")
                    print(
                        "Counts: "
                        f"perfect={score_state.perfect_count}, good={score_state.good_count}, "
                        f"ok={score_state.ok_count}, miss={score_state.miss_count}"
                    )
                    print(f"Skipped notes: {score_state.skipped_count}")
                    print(f"Extra hits: {score_state.extra_hits}")
                    print(f"Final offset: {score_state.offset_ms:+.1f}ms")
                    score_summary_printed = True

            cleanup_feedback(feedback_by_mid, wall_ms)

            # ------------------------------------------------------------
            # Banner text (still based on "next chord")
            # ------------------------------------------------------------
            cue_text = f"{title} | Press 's' to start"
            cue_ok = True
            next_group: List[dict] = []
            next_t: Optional[int] = None
            dt_next: Optional[int] = None

            if mode_blocked:
                cue_ok = False
                cue_text = f"{title} | mode={cfg.mode} | enable --stick-track to start"
            elif running:
                if cfg.mode == "play":
                    next_idx = advance_idx_past_old(events, next_idx, now_ms, cfg.hold_ms)
                    if next_idx >= len(events):
                        cue_text = f"{title} | DONE"
                    else:
                        next_t, next_group = group_chord(events, next_idx, cfg.chord_eps_ms)
                        dt_next = next_t - now_ms
                        next_names = [cfg.drum_id_to_name.get(int(e["drum"]), f"DRUM {e['drum']}") for e in next_group]
                        cue_text = f"{title} | NEXT: {' + '.join(next_names)} | in {max(dt_next, 0)}ms"
                elif cfg.mode == "train":
                    if train_state is None:
                        cue_text = f"{title} | mode=train"
                    elif train_state.done:
                        cue_text = (
                            f"{title} | mode=train | DONE | "
                            f"hits {train_state.correct_hits} | mistakes {train_state.mistakes}"
                        )
                    else:
                        remaining_names = [marker_id_to_name.get(mid, f"ID {mid}") for mid in sorted(train_state.remaining_mids)]
                        next_text = " + ".join(remaining_names) if remaining_names else "NONE"
                        cue_text = (
                            f"{title} | mode=train | NEXT: {next_text} | "
                            f"chord {train_state.chord_number}/{max(1, train_state.total_chords)} | "
                            f"hits {train_state.correct_hits} | mistakes {train_state.mistakes}"
                        )
                elif cfg.mode == "score" and score_state is not None:
                    cue_text = update_score_mode_ui(title, score_state, running=True, now_ms=now_ms)
            elif cfg.mode == "train" and train_state is not None and not train_state.done:
                remaining_names = [marker_id_to_name.get(mid, f"ID {mid}") for mid in sorted(train_state.remaining_mids)]
                next_text = " + ".join(remaining_names) if remaining_names else "NONE"
                cue_text = (
                    f"{title} | mode=train | Press 's' to start | "
                    f"NEXT: {next_text} | chord {train_state.chord_number}/{max(1, train_state.total_chords)}"
                )
            elif cfg.mode == "score" and score_state is not None:
                cue_text = update_score_mode_ui(title, score_state, running=False, now_ms=now_ms)

            if (not running) and (auto_start_at_t is not None) and (not mode_blocked):
                sec_left = max(0, int(np.ceil(auto_start_at_t - loop_t)))
                cue_text = f"{title} | mode={cfg.mode} | AUTO START IN {sec_left}s"

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

                    # Train target highlight for remaining expected drums.
                    if (
                        running
                        and cfg.mode == "train"
                        and train_state is not None
                        and (mid in train_state.remaining_mids)
                    ):
                        draw_projected_ring_safe(
                            frame, mtx, dist, rvec, tvec,
                            drum_r_m * 1.08,
                            cfg.circle_pts,
                            cfg.min_circle_pts,
                            color=(0, 255, 255),
                            alpha=0.85,
                            thickness=max(cfg.base_ring_thickness + 2, 8),
                            fill_alpha=0.06,
                        )

                # Render pulses for this marker, if any
                q = pulses_by_mid.get(mid)
                if running and cfg.mode in ("play", "score") and q and has_pose:
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
                fb = feedback_by_mid.get(mid)
                if has_pose and fb is not None:
                    fb_color, fb_expires = fb
                    if wall_ms <= fb_expires:
                        rvec, tvec = pose
                        draw_projected_ring_safe(
                            frame, mtx, dist, rvec, tvec,
                            drum_r_m * 1.03,
                            cfg.circle_pts,
                            cfg.min_circle_pts,
                            color=fb_color,
                            alpha=0.95,
                            thickness=max(cfg.base_ring_thickness + 4, 8),
                            fill_alpha=0.22,
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

            if cfg.enable_stick_tracking and cfg.stick_debug:
                if stick_tip_full is not None:
                    tip_xy = np.asarray(stick_tip_full, dtype=np.float32).reshape(2)
                    tip_u, tip_v = int(round(float(tip_xy[0]))), int(round(float(tip_xy[1])))
                    cv2.circle(frame, (tip_u, tip_v), 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (tip_u, tip_v), 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)

                dbg_text = "tip: none"
                if closest_stick_info is not None:
                    dbg_mid, dbg_z, dbg_r, dbg_vr = closest_stick_info
                    dbg_name = marker_id_to_name.get(dbg_mid, f"ID {dbg_mid}")
                    dbg_text = f"{dbg_name} z={dbg_z:+.3f} r={dbg_r:.3f} vr={dbg_vr:+.2f}"
                cv2.putText(
                    frame, dbg_text, (16, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                )
                if stick_mask_debug is not None:
                    cv2.imshow("stick_mask", stick_mask_debug)

            # ------------------------------------------------------------
            # Banner warning if expected marker isn't visible
            # Uses pulse queues rather than only the next chord
            # ------------------------------------------------------------
            if running:
                if cfg.mode == "play" and pulses_by_mid:
                    missing_names: List[str] = []
                    for mid, q in pulses_by_mid.items():
                        if mid in poses:
                            continue
                        if marker_is_expected_soon(q, now_ms, cfg):
                            missing_names.append(marker_id_to_name.get(mid, f"ID {mid}"))
                    if missing_names:
                        cue_ok = False
                        cue_text = cue_text + f" | NO 3D POSE: {', '.join(sorted(set(missing_names)))}"
                elif cfg.mode == "train" and train_state is not None and train_state.remaining_mids:
                    missing_names = [
                        marker_id_to_name.get(mid, f"ID {mid}")
                        for mid in sorted(train_state.remaining_mids)
                        if mid not in poses
                    ]
                    if missing_names:
                        cue_ok = False
                        cue_text = cue_text + f" | NO 3D POSE: {', '.join(missing_names)}"

            draw_banner(frame, cue_text, ok=cue_ok)
            if (
                running
                and cfg.mode == "score"
                and score_state is not None
                and score_state.milestone_text
                and now_ms <= score_state.milestone_until_ms
            ):
                draw_center_popup(frame, score_state.milestone_text, score_state.milestone_color)
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
                if mode_blocked:
                    print(f"{cfg.mode.upper()} MODE WARNING: start blocked. Re-run with --stick-track.")
                    continue
                start_session(from_auto=False)
            if key == ord("r"):
                running = False
                start_t = None
                stick_tip_full = None
                reset_mode_state()
                print("Reset")
            if key == ord("n") and cfg.mode == "train" and running and train_state is not None:
                skip_train_target(train_state, events, cfg)
                print("Train: skipped to next chord")
    finally:
        capture.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
