import requests
import face_recognition
from multiprocessing import Queue, Process
from datetime import datetime
import cv2
import time


class FaceRecognizer:
    def __init__(self, area: str, subarea: str, tenant_id: str):
        self.__area = area
        self.__subarea = subarea
        self.__tenant_id = tenant_id
        self.__face_encodings = []
        self.__image_queue = Queue()
        self.__image_sender_process = Process(
            target=self.process_images, daemon=True)
        self.__image_sender_process.start()

    def __saveImageLocal(self, face_frame, i: int) -> str:
        now = datetime.now()
        objectName = f"faces/{self.__area}_{self.__subarea}_{now.strftime('%Y%m%d_%H%M%S')}{i}_{self.__tenant_id}.jpg"
        try:
            cv2.imwrite(objectName, face_frame)
            print("Face saved: " + objectName)
        except Exception as e:
            print("Rostro vacio o incompleto")

        return objectName

    def sendImageToS3(self, image_path) -> None:
        Api_Url = 'https://v9buc4do9f.execute-api.us-east-1.amazonaws.com/dev'
        bucket = 'smart-cam-images'
        image_name = image_path[image_path.rfind("/") + 1:]
        print(f"Nombre de la imagen: {image_name}")

        url_put = f"{Api_Url}/{bucket}/{image_name}"

        with open(image_path, 'rb') as file:
            image_data = file.read()
            headers_put = {'Content-Type': 'image/jpg'}
            # Realiza la solicitud PUT
            r = requests.put(url_put, data=image_data, headers=headers_put)
            print(f"Estado de la respuesta de subida: {r.status_code}")

        if r.status_code == 200:
            time.sleep(2)
            url_get = f"{Api_Url}/reo"
            params = {'objectKey': f"{image_name}"}

            headers_get = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            }

            r2 = requests.get(url_get, headers=headers_get, params=params)

            if r2.status_code == 200:
                response_data = r2.json()
                print(response_data)
            else:
                print(f"Estado de la respuesta de GET: {r2.status_code}")

    def process_images(self):
        while True:
            data = self.__image_queue.get()
            if data is None:
                break
            frame, known_face_locations, i = data
            face_encoding = face_recognition.face_encodings(
                frame, known_face_locations)[0]

            result = face_recognition.compare_faces(
                self.__face_encodings, face_encoding)

            print(f"Result: {result}")

            if True not in result:
                self.__face_encodings.append(face_encoding)
                face_frame = frame[known_face_locations[0][0]:known_face_locations[0][2],
                                   known_face_locations[0][3]:known_face_locations[0][1]]
                image_path = self.__saveImageLocal(face_frame, i)
                print(f"Se ha agregado una nueva cara a la base de datos")
                self.sendImageToS3(image_path)
            else:
                print(f"La cara ya existe en la base de datos")

    def addFaceEncoding(self, frame, known_face_locations: list, i: int) -> None:
        self.__image_queue.put((frame, known_face_locations, i))

    def stop(self):
        self.__image_queue.put(None)
        self.__image_sender_process.join()
        self.__image_sender_process.kill()
