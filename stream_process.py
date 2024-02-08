import cv2
import json
import os
from datetime import datetime
import threading
from queue import Queue
from facenet_pytorch import MTCNN


class Camera:
    def __init__(self, area, subarea, videoStreamUrl, s3Bucket, isFisheye, frameCaptureThreshold, tenant_id):
        self.area = area
        self.subarea = subarea
        self.videoStreamUrl = videoStreamUrl
        self.s3Bucket = s3Bucket
        self.isFisheye = isFisheye
        self.frameCaptureThreshold = frameCaptureThreshold
        self.tenant_id = tenant_id
        self.current_faces = 0
        self.mtcnn = MTCNN()

    def captureVideo(self, frame_queue):
        cap = cv2.VideoCapture(self.videoStreamUrl)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_queue.put(frame)
            else:
                break
        cap.release()

    def processFrame(self):
        frame_queue = Queue(maxsize=10)
        threading.Thread(target=self.captureVideo, args=(
            frame_queue,), daemon=True).start()
        frame_count = 0
        scale = 100  # Escala reducida para un procesamiento más rápido
        while True:
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            # Reducir la resolución del frame
            height, width, _ = frame.shape
            frame = cv2.resize(
                frame, (int(width * scale / 100), int(height * scale / 100)))
            # Detectar rostros cada 5 frames para reducir la carga de procesamiento
            if frame_count % 5 == 0:
                boxes, _ = self.mtcnn.detect(frame)
            # Dibujar un rectángulo alrededor de los rostros y guardar solo el rostro
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_image = frame[y1:y2, x1:x2]
                    if frame_count % self.frameCaptureThreshold == 0 and len(boxes) > 0 and len(boxes) != self.current_faces:
                        self.current_faces = len(boxes)
                        now = datetime.now()
                        objectName = f"images/{self.area}_{now.strftime('%Y%m%d_%H%M%S')}_{
                            self.tenant_id}.jpg"
                        if not os.path.exists('images'):
                            os.makedirs('images')
                        if cv2.imwrite(objectName, face_image):
                            print("Face saved: " + objectName)
            if boxes is None:
                self.current_faces = 0
            cv2.imshow('Frame', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


# Cargar configuración desde JSON
with open("config.json", "r") as conf:
    config = json.load(conf)

cam = Camera(config["area"], config["subarea"], config["videoStreamTest"], config["s3Bucket"],
             config["isFisheye"], config["frameCaptureThreshold"], config["tenant_id"])
try:
    cam.processFrame()
except Exception as e:
    print("Error: {}.".format(e))
