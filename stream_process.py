import json
import cv2
from datetime import datetime


class Camera:
    def __init__(self, area, subarea, videoStreamUrl, s3Bucket, isFisheye, frameCaptureThreshold):
        self.area = area
        self.subarea = subarea
        self.videoStreamUrl = videoStreamUrl
        self.s3Bucket = s3Bucket
        self.isFisheye = isFisheye
        self.frameCaptureThreshold = frameCaptureThreshold


def processFrame(camera):
    cap = cv2.VideoCapture(camera.videoStreamTest)
    if not cap.isOpened():
        print("No se pudo abrir el stream de video.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break
        frame_count += 1

        if frame_count % camera.frameCaptureThreshold == 0:
            now = datetime.now()
            objectName = f"frame_{now.strftime('%Y%m%d_%H%M%S')}.jpg"

            if cv2.imwrite(objectName, frame):
                print("Frame saved: " + objectName)
            else:
                print("NO SE GUARDO NADA")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Load configuration from JSON
with open("config.json", "r") as conf:
    config = json.load(conf)
    cam = Camera(config["area"], config["subarea"], config["videoStreamUrl"], config["s3Bucket"], config["isFisheye"],
                 config["frameCaptureThreshold"])

    try:
        processFrame(cam)
    except Exception as e:
        print("Error: {}.".format(e))
