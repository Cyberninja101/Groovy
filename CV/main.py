import cv2
import numpy as np

LABELS = {
    7: "SNARE",
    12: "HIHAT",
    3: "RIDE",
}

MARKER_LENGTH = 0.05  # meters, set to your real marker size

def draw_cube(frame, mtx, dist, rvec, tvec, size):
    # Cube centered on marker, base on marker plane (z = 0), height goes toward camera (z = +size)
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

    # base
    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[(i + 1) % 4]), (0, 255, 0), 2)

    # top
    for i in range(4, 8):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[4 + (i + 1 - 4) % 4]), (0, 255, 0), 2)

    # vertical edges
    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i + 4]), (0, 255, 0), 2)

def main():
    # Load camera calibration
    calib = np.load("calib.npz")
    mtx = calib["mtx"]
    dist = calib["dist"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            ids_flat = ids.flatten().tolist()
            print("Detected ids:", ids_flat)

            aruco.drawDetectedMarkers(frame, corners, ids)

            # Pose estimation
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, mtx, dist)

            for c, mid, rvec, tvec in zip(corners, ids.flatten(), rvecs, tvecs):
                mid = int(mid)
                rvec = rvec.reshape(3, 1)
                tvec = tvec.reshape(3, 1)

                # Draw axis (optional but helpful)
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_LENGTH * 0.5)

                # Draw cube
                draw_cube(frame, mtx, dist, rvec, tvec, MARKER_LENGTH)

                # Label text near marker
                pts = c.reshape(-1, 2)
                x, y = int(pts[0][0]), int(pts[0][1])
                label = LABELS.get(mid, f"ID {mid}")
                cv2.putText(frame, label, (x, max(30, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("aruco", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()