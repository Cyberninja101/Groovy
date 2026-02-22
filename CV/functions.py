import cv2
import numpy as np
from constants import *
from cam_constants import INTRINSIC_MAT, DISTORTION_COEFFS
import pyapriltags

def getDetector():
    """Returns an Apriltag Detector """
    aprilobj = pyapriltags.Detector(
        nthreads=NTHREADS, 
        quad_decimate=QUAD_DECIMATE, 
        quad_sigma = QUAD_SIGMA,
        refine_edges = REFINE_EDGES,
        decode_sharpening= DECODE_SHARPENING
        )
    return aprilobj

def getDetections(detector:pyapriltags.Detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detect(gray)


def drawDetections(frame, detections:list[pyapriltags.Detection]):
    for detection in detections:
        corners = detection.corners.astype(int)
        for i in range(4):
            cv2.line(frame, (corners[i][0], corners[i][1]), (corners[(i+1)%4][0], corners[(i+1)%4][1]), (0, 255, 0), 3)
            cv2.circle(frame, (corners[i][0], corners[i][1]), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(detection.tag_id), (corners[0][0] + 10, corners[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return frame

def getTagPose(detection:pyapriltags.Detection):
    """Treat the center of the tag as the world origin. Solve for the camera pose in that tag-frame, then invert it"""
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints = np.array([
            [-TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, 0],
            [TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, 0],
            [TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, 0],
            [-TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, 0]
        ]),
        imagePoints = detection.corners,
        cameraMatrix = INTRINSIC_MAT,
        distCoeffs = DISTORTION_COEFFS
    )

    rotationmat, _ = cv2.Rodrigues(rvec) #Convert the rotation vector to a rotation matrix.

    tvec = (np.array(tvec))
    rvec = (np.array(rvec))  

    return ok, rvec, tvec, rotationmat


def drawPose(frame, rvec, tvec):  #chatgpt wrote this function, not sure if its entirely right but it seems legit? theres some clipping when u turn it a certain way
    axis_length = 0.05
    axis_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, INTRINSIC_MAT, DISTORTION_COEFFS)

    corner = tuple(imgpts[0].ravel().astype(int))
    frame = cv2.line(frame, corner, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 5) #X-axis in blue
    frame = cv2.line(frame, corner, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 5) #Y-axis in green
    frame = cv2.line(frame, corner, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 5) #Z-axis in blue

    #project box:
    box_points = np.float32([
        [-TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, 0],
        [TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, 0],
        [TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, 0],
        [-TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, 0],
        [-TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH],
        [TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH],
        [TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH],
        [-TAG_SIDE_LENGTH/2, TAG_SIDE_LENGTH/2, -TAG_SIDE_LENGTH]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(box_points, rvec, tvec, INTRINSIC_MAT, DISTORTION_COEFFS)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    #Draw the box edges
    for i in range(4):
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[(i+1)%4]), (255, 255, 0), 2) #Base square in cyan
        frame = cv2.line(frame, tuple(imgpts[i+4]), tuple(imgpts[((i+1)%4)+4]), (255, 255, 0), 2) #Top square in cyan
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (255, 255, 0), 2) #Vertical edges in cyan



    tvec_text = f"tvec: [{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}] m"
    frame = cv2.putText(frame, tvec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame