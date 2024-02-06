import requests
import glob
import os

path = '..\\SmartCam\\images'
os.chdir(path)

def client():
    pattern = "*.jpg"
    files = glob.glob(pattern)

    if files:

        most_recent_file = max(files, key=os.path.getmtime)
        print(f"El archivo más reciente es: {most_recent_file}")

        Api_Url = 'https://v9buc4do9f.execute-api.us-east-1.amazonaws.com/dev'
        bucket = 'smart-cam-images'
        image_name = most_recent_file

        url_put = f"{Api_Url}/{bucket}/{image_name}"

        with open(most_recent_file, 'rb') as file:
            image_data = file.read()
            headers_put = {'Content-Type': 'image/jpg'}
            # Realiza la solicitud PUT
            r = requests.put(url_put, data=image_data, headers=headers_put)
            print(f"Estado de la respuesta de subida: {r.status_code}")

        url_get = f"{Api_Url}/reo"
        params = {'objectKey': f"{most_recent_file}"}

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
    else:
        print("No se encontraron archivos que coincidan con el patrón especificado.")


def file_size():
   return len(os.listdir(os.getcwd()))


number_files = file_size()
while True:
    if number_files != file_size():
        client()
        number_files = file_size()
