import cv2
import functions

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = functions.getDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        detections = functions.getDetections(detector, frame)

        if detections:
            functions.drawDetections(frame, detections)
            tagPook, rvec, tvec, rotationmat = functions.getTagPose(detections[0]) #rn js take the first one we see, but we could loop through them if we wanted
            functions.drawPose(frame, rvec, tvec)
        
        cv2.imshow("AprilTag Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break