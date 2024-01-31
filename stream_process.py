import cv2
import json
import face_recognition
import os
from datetime import datetime
import threading
from queue import Queue

class Camera:
    def __init__(self, area, subarea, videoStreamUrl, s3Bucket, isFisheye, frameCaptureThreshold , tenant_id):
        self.area = area
        self.subarea = subarea
        self.videoStreamUrl = videoStreamUrl
        self.s3Bucket = s3Bucket
        self.isFisheye = isFisheye
        self.frameCaptureThreshold = frameCaptureThreshold
        self.tenant_id = tenant_id

def captureVideo(camera, frame_queue):
    cap = cv2.VideoCapture(camera.videoStreamUrl)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        else:
            break
    cap.release()

def processFrame(camera: Camera):
    frame_queue = Queue(maxsize=10)
    threading.Thread(target=captureVideo, args=(camera, frame_queue), daemon=True).start()

    frame_count = 0
    scale = 100  # Escala reducida para un procesamiento más rápido

    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()

        # Reducir la resolución del frame
        height, width, _ = frame.shape
        frame = cv2.resize(frame, (int(width * scale / 100), int(height * scale / 100)))

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros cada 5 frames para reducir la carga de procesamiento
        if frame_count % 5 == 0:
            face_locations = face_recognition.face_locations(gray)

        # Dibujar un rectángulo alrededor de los rostros y guardar solo el rostro
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face_image = frame[top:bottom, left:right]

            if frame_count % camera.frameCaptureThreshold == 0 and len(face_locations) > 0:
                now = datetime.now()
                objectName = f"images/{camera.area}_{now.strftime('%Y%m%d_%H%M%S')}_{camera.tenant_id}.jpg"
                if not os.path.exists('images'):
                    os.makedirs('images')
                if cv2.imwrite(objectName, face_image):
                    print("Face saved: " + objectName)

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

cam = Camera(config["area"], config["subarea"], config["videoStreamUrl"],
             config["s3Bucket"], config["isFisheye"], config["frameCaptureThreshold"] , config["tenant_id"])

try:
    processFrame(cam)
except Exception as e:
    print("Error: {}.".format(e))
