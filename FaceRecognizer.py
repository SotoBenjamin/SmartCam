import requests
import face_recognition


class FaceRecognizer:
    def __init__(self, area: str, tenant_id: str):
        self.__area = area
        self.__tenant_id = tenant_id
        self.__face_encodings = []

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

    def addFaceEncoding(self, frame, known_face_locations: list, image_path: str) -> None:
        # face_encoding = face_recognition.face_encodings(frame, known_face_locations)[0]
        # self.__face_encodings.append(face_encoding)
        # print("Caras actuales: " + str(len(self.__face_encodings)))
        self.sendImageToS3(image_path)
