import cv2
import numpy as np
import pyapriltags

from constants import *
from cam_constants import INTRINSIC_MAT, DISTORTION_COEFFS

# -----------------------
# Precomputed constants
# -----------------------

# Camera params for pyapriltags pose (no distortion in this API)
# pyapriltags expects [fx, fy, cx, cy]. :contentReference[oaicite:1]{index=1}
CAMERA_PARAMS = [
    float(INTRINSIC_MAT[0, 0]),
    float(INTRINSIC_MAT[1, 1]),
    float(INTRINSIC_MAT[0, 2]),
    float(INTRINSIC_MAT[1, 2]),
]

_HALF = float(TAG_SIDE_LENGTH) / 2.0

# 3D tag corner model for solvePnP (cached, no per-frame allocation)
OBJ_POINTS = np.array(
    [
        [-_HALF, -_HALF, 0.0],
        [ _HALF, -_HALF, 0.0],
        [ _HALF,  _HALF, 0.0],
        [-_HALF,  _HALF, 0.0],
    ],
    dtype=np.float32,
)

# Pose drawing points (cached)
AXIS_LENGTH = 0.05
AXIS_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [AXIS_LENGTH, 0.0, 0.0],
        [0.0, AXIS_LENGTH, 0.0],
        [0.0, 0.0, -AXIS_LENGTH],
    ],
    dtype=np.float32,
)

# Box "height" in tag coordinates (change sign if you prefer the box to go the other way)
BOX_HEIGHT = float(TAG_SIDE_LENGTH)

BOX_POINTS = np.array(
    [
        [-_HALF, -_HALF, 0.0],
        [ _HALF, -_HALF, 0.0],
        [ _HALF,  _HALF, 0.0],
        [-_HALF,  _HALF, 0.0],
        [-_HALF, -_HALF, -BOX_HEIGHT],
        [ _HALF, -_HALF, -BOX_HEIGHT],
        [ _HALF,  _HALF, -BOX_HEIGHT],
        [-_HALF,  _HALF, -BOX_HEIGHT],
    ],
    dtype=np.float32,
)

# Project once for both axis + box (fewer OpenCV calls)
POSE_DRAW_POINTS = np.vstack([AXIS_POINTS, BOX_POINTS]).astype(np.float32)
AXIS_SLICE = slice(0, 4)
BOX_SLICE = slice(4, 12)

# If available, this can be more stable for planar square targets
_PNP_FLAG = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)


# -----------------------
# Detector + detection
# -----------------------

def getDetector():
    """Returns an AprilTag detector."""
    return pyapriltags.Detector(
        nthreads=NTHREADS,
        quad_decimate=QUAD_DECIMATE,
        quad_sigma=QUAD_SIGMA,
        refine_edges=REFINE_EDGES,
        decode_sharpening=DECODE_SHARPENING,
        # If you have DEBUG in constants, keep it 0 for speed
        # debug=0,
    )


def getDetections(detector: pyapriltags.Detector, frame, estimate_pose: bool = False):
    """
    Returns list of detections.
    If estimate_pose=True, pyapriltags will also compute pose_R and pose_t. :contentReference[oaicite:2]{index=2}
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame  # already grayscale

    if estimate_pose:
        return detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=float(TAG_SIDE_LENGTH),
        )
    else:
        return detector.detect(gray)


# -----------------------
# Drawing
# -----------------------

def drawDetections(frame, detections):
    """
    Faster drawing: use polylines instead of 4 separate line calls.
    """
    for det in detections:
        corners = det.corners.astype(np.int32)  # (4,2)
        poly = corners.reshape((-1, 1, 2))
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

        for (x, y) in corners:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        x0, y0 = int(corners[0, 0]), int(corners[0, 1])
        cv2.putText(
            frame,
            str(det.tag_id),
            (x0 + 10, max(30, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


# -----------------------
# Pose
# -----------------------

def getTagPose(
    detection,
    prefer_detector_pose: bool = True,
):
    """
    Returns (ok, rvec, tvec, rotationmat)

    Fast path: if the detection already has pose_R/pose_t (from estimate_pose=True),
    use those and skip solvePnP. :contentReference[oaicite:3]{index=3}

    Accuracy path: solvePnP uses distortion coefficients.
    """
    # Fast path: pose computed by pyapriltags
    if prefer_detector_pose and hasattr(detection, "pose_R") and hasattr(detection, "pose_t"):
        R = np.asarray(detection.pose_R, dtype=np.float32)
        t = np.asarray(detection.pose_t, dtype=np.float32).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        return True, rvec.astype(np.float32), t, R

    img_points = np.asarray(detection.corners, dtype=np.float32)

    if _PNP_FLAG is not None:
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=OBJ_POINTS,
            imagePoints=img_points,
            cameraMatrix=INTRINSIC_MAT,
            distCoeffs=DISTORTION_COEFFS,
            flags=_PNP_FLAG,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=OBJ_POINTS,
            imagePoints=img_points,
            cameraMatrix=INTRINSIC_MAT,
            distCoeffs=DISTORTION_COEFFS,
        )

    R, _ = cv2.Rodrigues(rvec)
    return bool(ok), rvec.astype(np.float32), np.asarray(tvec, dtype=np.float32), R.astype(np.float32)


def drawPose(frame, rvec, tvec):
    """
    Faster: project axis + box in one cv2.projectPoints call.
    """
    rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float32).reshape(3, 1)

    imgpts, _ = cv2.projectPoints(
        POSE_DRAW_POINTS,
        rvec,
        tvec,
        INTRINSIC_MAT,
        DISTORTION_COEFFS,
    )
    imgpts = imgpts.reshape(-1, 2).astype(np.int32)

    axis = imgpts[AXIS_SLICE]
    box = imgpts[BOX_SLICE]

    origin = tuple(axis[0])
    # BGR colors: X red, Y green, Z blue
    cv2.line(frame, origin, tuple(axis[1]), (0, 0, 255), 4)
    cv2.line(frame, origin, tuple(axis[2]), (0, 255, 0), 4)
    cv2.line(frame, origin, tuple(axis[3]), (255, 0, 0), 4)

    # Box edges
    for i in range(4):
        cv2.line(frame, tuple(box[i]), tuple(box[(i + 1) % 4]), (255, 255, 0), 2)
        cv2.line(frame, tuple(box[i + 4]), tuple(box[((i + 1) % 4) + 4]), (255, 255, 0), 2)
        cv2.line(frame, tuple(box[i]), tuple(box[i + 4]), (255, 255, 0), 2)

    tvec_text = f"tvec: [{tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f}] m"
    cv2.putText(frame, tvec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame