import json
import cv2
from queue import Queue
import threading
import mediapipe as mp
from FaceRecognizer import FaceRecognizer
import os
from datetime import datetime


class Camera:
    def __init__(self, config_json, isTest=False):
        with open(config_json, 'r') as f:
            config = json.load(f)

        # Camera Properties
        self.__area = config["area"]
        self.__subarea = config["subarea"]
        self.__videoStreamUrl = 0 if isTest else config["videoStreamUrl"]
        self.__s3Bucket = config["s3Bucket"]
        self.__isFishEye = config["isFishEye"]
        self.__frameCaptureThreshold = config["frameCaptureThreshold"]
        self.__tenant_id = config["tenant_id"]
        self.__fps = config["fps"]
        self.__width_res = config["width_res"]
        self.__height_res = config["height_res"]

        # Execution Properties
        self.__frame_queue = Queue(maxsize=20)
        self.__thread = threading.Thread(
            target=self.__captureVideo, daemon=True)
        self.__frame_count = 0
        self.__scale = 50
        self.__results = None
        self.__current_faces = 0
        self.__face_recognizer = FaceRecognizer(self.__area, self.__tenant_id)
        self.__face_detection = mp.solutions.face_detection.FaceDetection()

    def __captureVideo(self):
        cap = cv2.VideoCapture(self.__videoStreamUrl)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.__fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width_res)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height_res)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.__frame_queue.put(frame)
            else:
                break

    def __drawBorders(self, frame, x1, y1, w, h) -> None:
        line_size = 20
        color = (255, 255, 0)
        thickness = 2

        # Superior izquierda
        cv2.line(frame, (x1, y1), (x1 + line_size, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + line_size), color, thickness)

        # Superior derecha
        cv2.line(frame, (x1 + w, y1), (x1 + w -
                 line_size, y1), color, thickness)
        cv2.line(frame, (x1 + w, y1),
                 (x1 + w, y1 + line_size), color, thickness)

        # Inferior izquierda
        cv2.line(frame, (x1, y1 + h),
                 (x1 + line_size, y1 + h), color, thickness)
        cv2.line(frame, (x1, y1 + h), (x1, y1 +
                 h - line_size), color, thickness)

        # Inferior derecha
        cv2.line(frame, (x1 + w, y1 + h), (x1 + w -
                 line_size, y1 + h), color, thickness)
        cv2.line(frame, (x1 + w, y1 + h),
                 (x1 + w, y1 + h - line_size), color, thickness)

    def __saveImageLocal(self, face_frame, i: int) -> str:
        now = datetime.now()
        objectName = f"faces/{self.__area}_{self.__subarea}_{now.strftime('%Y%m%d_%H%M%S')}{i}_{self.__tenant_id}.jpg"
        try:
            cv2.imwrite(objectName, face_frame)
            print("Face saved: " + objectName)
        except Exception as e:
            print("Rostro vacio o incompleto")

        return objectName

    def start(self):
        if not os.path.exists('faces'):
            os.makedirs('faces')
        else:
            for file in os.listdir('faces'):
                os.remove(f"faces/{file}")
        try:
            self.__thread.start()
            while True:
                if self.__frame_queue.empty():
                    continue
                frame = self.__frame_queue.get()

                if frame is None:
                    continue

                height, width, _ = frame.shape
                frame = cv2.resize(
                    frame, (int(width * self.__scale / 100), int(height * self.__scale / 100)))

                if self.__frame_count % 5 == 0:
                    self.__frame_count = 0
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.__results = self.__face_detection.process(frame_rgb)

                if self.__results.detections:
                    for i, face in enumerate(self.__results.detections):
                        # Obtener el cuadro delimitador
                        x1, y1, w, h = face.location_data.relative_bounding_box.xmin, face.location_data.relative_bounding_box.ymin, face.location_data.relative_bounding_box.width, face.location_data.relative_bounding_box.height
                        x1, y1, w, h = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(
                            w * frame.shape[1]), int(h * frame.shape[0])

                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        w = min(w, frame.shape[1] - x1)
                        h = min(h, frame.shape[0] - y1)

                        face_frame = frame[y1:y1 + h, x1:x1 + w]

                        if self.__current_faces != len(self.__results.detections):
                            objectName = self.__saveImageLocal(face_frame, i)
                            thread = threading.Thread(
                                target=self.__face_recognizer.addFaceEncoding, args=(frame, [(y1, x1 + w, y1 + h, x1)], objectName))
                            thread.start()

                        self.__drawBorders(frame, x1, y1, w, h)

                    self.__current_faces = len(self.__results.detections)
                else:
                    self.__current_faces = 0

                cv2.imshow('Frame', frame)
                self.__frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                    break

            cv2.destroyAllWindows()

        except Exception as e:
            print("Error: {}.".format(e))


if __name__ == "__main__":
    cam = Camera("config.json", True)
    cam.start()
