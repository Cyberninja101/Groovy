import json
import time
import cv2
import numpy as np

# ----------------------------
# CONFIG YOU MUST SET
# ----------------------------

SONG_PATH = "Audio/brandy-1350.json"  # put your json file here

# Song uses numeric drum IDs, map those to human names
DRUM_ID_TO_NAME = {
    1: "SNARE",
    3: "RIDE",
    5: "HIHAT",
    6: "CRASH",
}

# Map drum name -> marker id (ArUco ID on that drum)
# This is your "calibration" for now. Update to match your kit.
NAME_TO_MARKER_ID = {
    "SNARE": 7,
    "HIHAT": 12,
    "RIDE": 3,
    "CRASH": 5,
}

# Timing behavior
LEAD_MS = 250      # start highlighting this many ms before the hit time
HOLD_MS = 120      # keep highlight briefly after time passes (makes it readable)
CHORD_EPS_MS = 25  # if multiple events share nearly same t_ms, treat as same hit

# Performance
USE_3D_CUBE = False   # turn on later if you want, costs CPU
MARKER_LENGTH = 0.05  # meters, only needed if USE_3D_CUBE is True

# ----------------------------
# Utility
# ----------------------------

def load_song(path):
    with open(path, "r") as f:
        data = json.load(f)
    events = data["events"]
    events.sort(key=lambda e: e["t_ms"])
    return data.get("title", "song"), events

def group_next_events(events, idx, now_ms):
    """
    Returns (next_t_ms, grouped_events, new_idx)
    grouped_events are all events at ~the same timestamp (chord).
    idx should be advanced past events that are too far in the past.
    """
    # Advance idx past events that are way behind (older than HOLD window)
    while idx < len(events) and events[idx]["t_ms"] < now_ms - HOLD_MS:
        idx += 1
    if idx >= len(events):
        return None, [], idx

    next_t = events[idx]["t_ms"]
    group = [events[idx]]
    j = idx + 1
    while j < len(events) and abs(events[j]["t_ms"] - next_t) <= CHORD_EPS_MS:
        group.append(events[j])
        j += 1
    return next_t, group, idx

def draw_banner(frame, text, ok=True):
    h, w = frame.shape[:2]
    bar_h = 60
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    color = (0, 255, 0) if ok else (0, 255, 255)  # green if visible, yellow if missing
    cv2.putText(frame, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

def bbox_from_corners(c):
    pts = c.reshape(-1, 2)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# Optional 3D: only if you want later
def draw_cube(frame, mtx, dist, rvec, tvec, size):
    s = size / 2.0
    obj_pts = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0],
        [-s, -s, size],
        [ s, -s, size],
        [ s,  s, size],
        [-s,  s, size],
    ], dtype=np.float32)

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[(i + 1) % 4]), (0, 255, 0), 2)
    for i in range(4, 8):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[4 + (i + 1 - 4) % 4]), (0, 255, 0), 2)
    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i + 4]), (0, 255, 0), 2)

def main():
    title, events = load_song(SONG_PATH)
    print(f"Loaded song: {title}, events: {len(events)}")
    print("Controls: s = start, r = reset, q = quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # If you need speed, reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    # 3D calibration (optional)
    mtx = dist = None
    if USE_3D_CUBE:
        calib = np.load("calib.npz")
        mtx = calib["mtx"]
        dist = calib["dist"]

    running = False
    start_t = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        detected = {}  # marker_id -> corners
        if ids is not None:
            for c, mid in zip(corners, ids.flatten()):
                detected[int(mid)] = c

        # Time logic
        now_ms = 0
        if running and start_t is not None:
            now_ms = int((time.perf_counter() - start_t) * 1000)

        next_t, group, idx = group_next_events(events, idx, now_ms) if running else (None, [], idx)

        # Decide if we should highlight now
        highlight_markers = []
        cue_text = f"{title} | Press 's' to start"
        cue_ok = True

        if running:
            if next_t is None:
                cue_text = f"{title} | DONE"
            else:
                dt = next_t - now_ms  # ms until next hit time
                # If within lead window or slightly late (within hold), highlight
                if dt <= LEAD_MS and dt >= -HOLD_MS:
                    for e in group:
                        drum_id = e["drum"]
                        name = DRUM_ID_TO_NAME.get(drum_id, f"DRUM {drum_id}")
                        mid = NAME_TO_MARKER_ID.get(name)
                        if mid is not None:
                            highlight_markers.append((mid, name))
                # Banner text shows what's next either way
                next_names = []
                for e in group:
                    name = DRUM_ID_TO_NAME.get(e["drum"], f"DRUM {e['drum']}")
                    next_names.append(name)
                cue_text = f"{title} | NEXT: {' + '.join(next_names)} | in {max(dt,0)}ms"

        # Draw all detections as normal boxes
        for mid, c in detected.items():
            x, y, w, h = bbox_from_corners(c)
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Default: thin green
            color = (0, 255, 0)
            thickness = 2

            # If this marker is highlighted: thick red
            names_for_mid = [nm for (hm, nm) in highlight_markers if hm == mid]
            if names_for_mid:
                color = (0, 0, 255)
                thickness = 6

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # draw marker id
            cv2.putText(frame, f"ID {mid}", (x, max(30, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # draw drum name if we can infer it
            for nm in names_for_mid:
                cv2.putText(frame, nm, (x, y + h + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Banner behavior if highlighted marker not visible
        if running and highlight_markers:
            missing = [name for (mid, name) in highlight_markers if mid not in detected]
            if missing:
                cue_ok = False
                cue_text = cue_text + f" | NOT VISIBLE: {', '.join(missing)}"

        draw_banner(frame, cue_text, ok=cue_ok)

        cv2.imshow("drum_overlay", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            running = True
            start_t = time.perf_counter()
            idx = 0
            print("Song started")
        if key == ord("r"):
            running = False
            start_t = None
            idx = 0
            print("Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()