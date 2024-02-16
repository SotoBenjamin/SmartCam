from queue import Queue
from threading import Thread
from deepface import DeepFace
from datetime import datetime
import os
import cv2


class FaceRecognizer:
    def __init__(self):
        self.__faces_queue = Queue()
        self.__threads_available = os.cpu_count()
        self.__counter = 0
        self.__faces_registered = set()

    def add_face(self, frame):
        self.__faces_queue.put(frame)
        print("Caras en la cola: ", self.__faces_queue.qsize())
        if self.__counter < 3:
            self.__analize_face()
            self.__counter += 1

    def __analize_face(self):
        face = self.__faces_queue.get()

        if len(self.__faces_registered) == 0:
            now = datetime.now()
            objectName = f"faces/{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            if cv2.imwrite(objectName, face):
                print("Face saved: " + objectName)
                self.__faces_registered.add(objectName)
        else:
            for face_registered in self.__faces_registered:
                try:
                    result = DeepFace.verify(face, face_registered, model_name="VGG-Face",
                                             detector_backend="fastmtcnn", distance_metric="euclidean_l2")
                    print("Result: ", result)
                except Exception as e:
                    print("Error: ", e)
