import cv2
import time

# Inicializa la cámara o carga un video
cap = cv2.VideoCapture('cosa.mp4')  # Reemplaza 'cosa.mp4' con el nombre de tu archivo de video

# Verifica si la cámara o el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara o el video.")
    exit()

# Configura el intervalo de tiempo entre capturas (en segundos)
intervalo_tiempo = 500  # Cambia esto al intervalo de tiempo deseado

# Inicializa el contador de tiempo
tiempo_anterior = time.time()

# Inicializa el contador de frames
contador_frames = 0

while True:
    # Lee un frame del video
    ret, frame = cap.read()

    if not ret:
        print("Error al leer el frame.")
        break

    # Muestra el frame
    cv2.imshow('frame', frame)

    # Incrementa el contador de frames
    contador_frames = cap.get(1) #current frame
    print(contador_frames)
    # Obtiene el tiempo actual
    tiempo_actual = time.time()

    # Comprueba si ha pasado el intervalo de tiempo deseado
    if contador_frames % intervalo_tiempo == 0:
        # Guarda el frame actual
        nombre_archivo = f"frame_A_{contador_frames}.jpg"
        cv2.imwrite(nombre_archivo, frame)
        hasFrame, imageBytes = cv2.imencode(".jpg", frame)
        print(f"Frame {contador_frames} guardado como {nombre_archivo}")

        # Reinicia el contador de tiempo
        tiempo_anterior = tiempo_actual

    # Rompe el bucle si se presiona 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
