import cv2
import json
import os
from datetime import datetime
import threading
from queue import Queue
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import torchvision.transforms.functional as TF


class Camera:
    def __init__(self, area, subarea, videoStreamUrl, s3Bucket, isFisheye, frameCaptureThreshold, tenant_id, fps, width_res, height_res):
        # Configuración de la cámara
        self.fps = fps
        self.width_res = width_res
        self.height_res = height_res
        self.area = area
        self.subarea = subarea
        self.videoStreamUrl = videoStreamUrl
        self.s3Bucket = s3Bucket
        self.isFisheye = isFisheye
        self.frameCaptureThreshold = frameCaptureThreshold
        self.tenant_id = tenant_id
        # Configuración de la detección de rostros
        self.new_face_event = threading.Event()
        self.current_faces = 0
        self.rostros = set()
        self.threads_available = os.cpu_count()
        self.faces_queue = Queue()
        self.mtcnn = MTCNN()
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def compare_faces(self, embedding, rostro_guardado, umbral=1.2):
        distancia = torch.dist(embedding, rostro_guardado)
        return distancia < umbral

    def transform_image(self, image):
        image = cv2.resize(image, (160, 160))
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        image = image.unsqueeze(0)
        return image

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

    def processImage(self):
        while True:
            self.new_face_event.wait()
            while not self.faces_queue.empty():
                face_image = self.faces_queue.get()
                # Comparacion de rostros
                rostro_transformado = self.transform_image(face_image)
                embedding = self.facenet(rostro_transformado)
                # Compara el nuevo rostro con los rostros guardados
                for rostro_guardado in self.rostros:
                    if self.compare_faces(embedding, rostro_guardado):
                        break
                else:
                    self.rostros.add(embedding)
                    now = datetime.now()
                    objectName = f"images/{self.area}_{now.strftime('%Y%m%d_%H%M%S')}_{self.tenant_id}.jpg"
                    if not os.path.exists('images'):
                        os.makedirs('images')
                    if cv2.imwrite(objectName, face_image):
                        print("Face saved: " + objectName)
                print("Elementos restantes en la cola: " +
                      str(self.faces_queue.qsize()))
            self.new_face_event.clear()

    def processFrame(self):
        frame_queue = Queue(maxsize=25)
        threading.Thread(target=self.captureVideo, args=(
            frame_queue,), daemon=True).start()
        threading.Thread(target=self.processImage, daemon=True).start()
        frame_count = 0
        scale = 50  # Escala para mostrar en pantalla

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

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_image = frame[y1:y2, x1:x2]

                    if self.current_faces != len(boxes):
                        self.faces_queue.put(face_image)

                self.new_face_event.set()
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

cam = Camera(config["area"], config["subarea"], 0, config["s3Bucket"],
             config["isFishEye"], config["frameCaptureThreshold"], config["tenant_id"], config["fps"], config["width_res"], config["height_res"])
try:
    cam.processFrame()
except Exception as e:
    print("Error: {}.".format(e))
