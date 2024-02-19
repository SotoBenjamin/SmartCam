import json
import cv2
from queue import Queue
import threading
from facenet_pytorch import MTCNN
from FaceRecognizer import FaceRecognizer
import os


class Camera:
    def __init__(self, config_json, isTest=False):
        with open(config_json, 'r') as f:
            config = json.load(f)

        # Camera Properties
        self.__area = config["area"]
        self.__subarea = config["subarea"]
        self.__videoStreamUrl = 0 if isTest else config["videoStreamUrl"]
        self.__s3Bucket = config["s3Bucket"]
        self.__isFishEye = config["isFishEye"]
        self.__frameCaptureThreshold = config["frameCaptureThreshold"]
        self.__tenant_id = config["tenant_id"]
        self.__fps = config["fps"]
        self.__width_res = config["width_res"]
        self.__height_res = config["height_res"]

        # Execution Properties
        self.__frame_queue = Queue(maxsize=20)
        self.__thread = threading.Thread(
            target=self.__captureVideo, daemon=True)
        self.__frame_count = 0
        self.__scale = 50
        self.__results = None
        self.__current_faces = 0
        self.__face_recognizer = FaceRecognizer(self.__area, self.__tenant_id)
        self.__mtcnn = MTCNN(select_largest=False, keep_all=True)

    def __captureVideo(self):
        cap = cv2.VideoCapture(self.__videoStreamUrl)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.__fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width_res)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height_res)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.__frame_queue.put(frame)
            else:
                break

    def start(self):
        if not os.path.exists('faces'):
            os.makedirs('faces')
        else:
            for file in os.listdir('faces'):
                os.remove(f"faces/{file}")
        try:
            self.__thread.start()
            while True:
                if self.__frame_queue.empty():
                    continue
                frame = self.__frame_queue.get()

                if frame is None:
                    continue

                height, width, _ = frame.shape
                frame = cv2.resize(
                    frame, (int(width * self.__scale / 100), int(height * self.__scale / 100)))

                if self.__frame_count % 5 == 0:
                    self.__frame_count = 0
                    try:
                        # show a rectangle for each face
                        self.__results, _ = self.__mtcnn.detect(frame)
                    except Exception as e:
                        self.__results = None

                if self.__results is not None:
                    for box in self.__results:
                        x1, y1, x2, y2 = box.astype(int)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width-1, x2), min(height-1, y2)

                        if self.__current_faces != len(self.__results):
                            self.__face_recognizer.add_face(
                                frame[y1:y2, x1:x2])

                        # cv2.rectangle(frame, (x1, y1),(x2, y2), (255, 255, 0), 1)
                    self.__current_faces = len(self.__results)
                else:
                    self.__current_faces = 0

                # cv2.imshow('Frame', frame)
                self.__frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        except Exception as e:
            print("Error: {}.".format(e))


if __name__ == "__main__":
    cam = Camera("config.json", True)
    cam.start()
