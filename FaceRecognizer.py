import requests
import face_recognition
from multiprocessing import Queue, Process
import queue


class FaceRecognizer:
    def __init__(self, area: str, tenant_id: str):
        self.__area = area
        self.__tenant_id = tenant_id
        self.__face_encodings = []
        self.__image_queue = Queue()
        self.__image_sender_process = Process(
            target=self.process_images, daemon=True)
        self.__image_sender_process.start()

    def sendImageToS3(self, image_path):
        Api_Url = 'https://v9buc4do9f.execute-api.us-east-1.amazonaws.com/dev'
        bucket = 'smart-cam-images'
        image_name = image_path[image_path.rfind("/") + 1:]

        url_put = f"{Api_Url}/{bucket}/{image_name}"

        with open(image_path, 'rb') as file:
            image_data = file.read()
            headers_put = {'Content-Type': 'image/jpg'}
            # Realiza la solicitud PUT
            r = requests.put(url_put, data=image_data, headers=headers_put)
            print(f"Estado de la respuesta de subida: {r.status_code}")

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
            try:
                data = self.__image_queue.get(
                    timeout=1)  # Espera hasta 1 segundo
            except queue.Empty:
                continue
            if data is None:  # Señal de parada
                break
            frame, known_face_locations, image_path = data
            face_encoding = face_recognition.face_encodings(
                frame, known_face_locations)[0]
            self.__face_encodings.append(face_encoding)
            print("Caras actuales: " + str(len(self.__face_encodings)))
            self.sendImageToS3(image_path)

    def addFaceEncoding(self, frame, known_face_locations: list, image_path: str) -> None:
        self.__image_queue.put((frame, known_face_locations, image_path))

    def stop(self):
        self.__image_queue.put(None)
        self.__image_sender_process.join()
        self.__image_sender_process.close()
