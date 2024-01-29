import cv2
import json
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
    # Asegúrate de que este es el camino correcto al archivo xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0 and frame_count % camera.frameCaptureThreshold == 0:
            now = datetime.now()
            objectName = f"frame_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            if cv2.imwrite(objectName, resized_cropped):
                print("Frame saved: " + objectName)
            else:
                print("NO SE GUARDO NADA")

        # Dibujar un rectángulo alrededor de los rostros
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_cropped, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_count += 1

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
