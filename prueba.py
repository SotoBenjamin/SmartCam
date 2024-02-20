import cv2
import mediapipe as mp

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Inicializar MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Convertir el color de BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar las caras
    results = face_detection.process(image)

    # Dibujar las anotaciones de detección de cara en la imagen
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    # Mostrar la imagen
    cv2.imshow('MediaPipe Face Detection', image)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
