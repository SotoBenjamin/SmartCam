from multiprocessing import Process, Queue
from deepface import DeepFace
from datetime import datetime
import cv2


def worker(faces_queue: Queue, faces_registered, area, tenant_id):
    try:
        while True:
            if not faces_queue.empty():
                face = faces_queue.get()
                print("Caras en la cola: ", faces_queue.qsize())
                analize_face(face, faces_registered, area, tenant_id)
    finally:
        faces_queue.close()


def analize_face(face, faces_registered, area, tenant_id) -> None:

    for saved_face in faces_registered:
        result = DeepFace.verify(face, saved_face, enforce_detection=False,
                                 detector_backend="opencv", model_name="Dlib", distance_metric="euclidean_l2")
        if result["verified"]:
            print("Face is similar to a saved face, not saving.")
            return

    now = datetime.now()
    objectName = f"faces/{area}_{now.strftime('%Y%m%d_%H%M%S')}_{tenant_id}.jpg"
    cv2.imwrite(objectName, face)
    print("Face saved: " + objectName)
    faces_registered.add(objectName)

    return


class FaceRecognizer:
    def __init__(self, area: str, tenant_id: str):
        self.__faces_queue = Queue()
        self.__faces_registered = set()
        self.__area = area
        self.__tenant_id = tenant_id
        self.__worker_process = Process(
            target=worker, args=(self.__faces_queue, self.__faces_registered, self.__area, self.__tenant_id), daemon=True)
        self.__worker_process.start()

    def add_face(self, frame):
        self.__faces_queue.put(frame)

    def clear_queue(self):
        # Add a sentinel value to indicate the worker to stop processing
        self.__faces_queue.put(None)

        # Wait for the worker process to finish
        self.__worker_process.join()

        # Close the queue
        self.__faces_queue.close()
