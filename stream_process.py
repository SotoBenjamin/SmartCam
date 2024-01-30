import cv2
import json
import face_recognition
import os
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
    cap = cv2.VideoCapture(camera.videoStreamUrl)
    if not cap.isOpened():
        print("No se pudo abrir el stream de video.")
        return
    frame_count = 0
    scale = 500
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # obtener el tamaño de la webcam
        height, width, channels = frame.shape

        # preparar el recorte
        centerX, centerY = int(height/2), int(width/2)
        radiusX, radiusY = int(scale*height/100), int(scale*width/100)

        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY

        cropped = frame[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height))

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        face_locations = face_recognition.face_locations(gray)

        # Dibujar un rectángulo alrededor de los rostros
        for top, right, bottom, left in face_locations:
            cv2.rectangle(resized_cropped, (left-10, top-50),
                          (right+10, bottom+10), (0, 255, 0), 2)

        frame_count += 1
        if frame_count % camera.frameCaptureThreshold == 0 and len(face_locations) > 0:
            now = datetime.now()
            # Modificar esta línea
            objectName = f"images/frame_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            if not os.path.exists('images'):
                os.makedirs('images')
            if cv2.imwrite(objectName, resized_cropped):
                print("Frame saved: " + objectName)
            else:
                print("NO SE GUARDO NADA")
        cv2.imshow('Frame', resized_cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Load configuration from JSON
with open("config.json", "r") as conf:
    config = json.load(conf)

cam = Camera(config["area"], config["subarea"], config["videoStreamTest"],
             config["s3Bucket"], config["isFisheye"], config["frameCaptureThreshold"])

try:
    processFrame(cam)
except Exception as e:
    print("Error: {}.".format(e))
