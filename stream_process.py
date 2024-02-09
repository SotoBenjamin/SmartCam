import cv2
import json
import os
from datetime import datetime
import threading
from queue import Queue
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Camera:
    def __init__(self, area, subarea, videoStreamUrl, s3Bucket, isFisheye, frameCaptureThreshold, tenant_id, fps, width_res, height_res):
        self.area = area
        self.subarea = subarea
        self.videoStreamUrl = videoStreamUrl
        self.s3Bucket = s3Bucket
        self.isFisheye = isFisheye
        self.frameCaptureThreshold = frameCaptureThreshold
        self.tenant_id = tenant_id
        self.current_faces = 0
        self.mtcnn = MTCNN()
        self.rostros = []
        self.fps = fps
        self.width_res = width_res
        self.height_res = height_res

    def captureVideo(self, frame_queue):
        cap = cv2.VideoCapture(self.videoStreamUrl)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width_res)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height_res)

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
        scale = 50  # Escala reducida para un procesamiento más rápido

        facenet = InceptionResnetV1(pretrained='vggface2').eval()

        while True:
            if frame_queue.empty():
                continue
            frame = frame_queue.get()

            if frame is None:
                continue

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

                    if self.current_faces != len(boxes):
                        # Comparacion de rostros
                        rostro_rgb = cv2.cvtColor(
                            face_image, cv2.COLOR_BGR2RGB)
                        rostro_transformado = transform(rostro_rgb)

                        rostro_transformado = rostro_transformado.unsqueeze(0)
                        embedding = facenet(rostro_transformado)

                        # Compara el nuevo rostro con los rostros guardados
                        for rostro_guardado in self.rostros:
                            distancia = torch.dist(embedding, rostro_guardado)
                            if distancia < 1.3:
                                break
                        else:
                            # Si el rostro no está en la lista, lo añade
                            self.rostros.append(embedding)
                            now = datetime.now()
                            objectName = f"images/{self.area}_{now.strftime('%Y%m%d_%H%M%S')}_{
                                self.tenant_id}.jpg"
                            if not os.path.exists('images'):
                                os.makedirs('images')
                            if cv2.imwrite(objectName, face_image):
                                print("Face saved: " + objectName)

                self.current_faces = len(boxes)

            else:
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
             config["isFisheye"], config["frameCaptureThreshold"], config["tenant_id"], config["fps"], config["width_res"], config["height_res"])
try:
    cam.processFrame()
except Exception as e:
    print("Error: {}.".format(e))
