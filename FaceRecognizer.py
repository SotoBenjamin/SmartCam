from multiprocessing import Process, Queue
from deepface import DeepFace
from datetime import datetime
import cv2


def worker(faces_queue: Queue, faces_registered):
    while True:
        if not faces_queue.empty():
            face = faces_queue.get()
            print("Caras en la cola: ", faces_queue.qsize())
            analize_face(face, faces_registered)


def analize_face(face, faces_registered):
    for saved_face in faces_registered:
        result = DeepFace.verify(
            face, saved_face, enforce_detection=False, detector_backend="skip", model_name="DeepID")
        if result["verified"]:
            print("Face is similar to a saved face, not saving.")
            return

    # Si la cara actual no es similar a ninguna cara guardada, guárdala
    now = datetime.now()
    objectName = f"faces/{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(objectName, face)
    print("Face saved: " + objectName)
    faces_registered.add(objectName)


class FaceRecognizer:
    def __init__(self):
        self.__faces_queue = Queue()
        self.__faces_registered = set()
        self.__worker_process = Process(
            target=worker, args=(self.__faces_queue, self.__faces_registered), daemon=True).start()

    def add_face(self, frame):
        self.__faces_queue.put(frame)
