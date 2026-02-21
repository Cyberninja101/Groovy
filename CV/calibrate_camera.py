import cv2
import numpy as np
import glob

# Inner corners on your checkerboard, example 9x6 means 9 by 6 inner corners
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.024  # meters, set this to your real square size

def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press SPACE to capture a calibration frame when corners are visible.")
    print("Press Q to finish and calibrate.")

    last_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_gray = gray

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        vis = frame.copy()

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)

        cv2.imshow("calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured {len(objpoints)} frames")
            else:
                print("No corners found, try again")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError("Need at least ~10 good frames for decent calibration")

    h, w = last_gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    print("RMS reprojection error:", ret)
    np.savez("calib.npz", mtx=mtx, dist=dist, w=w, h=h)
    print("Saved calib.npz")

if __name__ == "__main__":
    main()