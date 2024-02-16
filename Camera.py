import json
import cv2
from queue import Queue
import threading
from deepface import DeepFace


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
        self.__frame_queue = Queue(maxsize=30)
        self.__thread = threading.Thread(
            target=self.__captureVideo, daemon=True)
        self.__frame_count = 0
        self.__scale = 50
        self.__results = None

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

        try:
            self.__thread.start()
            while True:
                if self.__frame_queue.empty():
                    continue
                frame = self.__frame_queue.get()

                height, width, _ = frame.shape
                frame = cv2.resize(
                    frame, (int(width * self.__scale / 100), int(height * self.__scale / 100)))

                if self.__frame_count % 5 == 0:
                    try:
                        self.__results = DeepFace.extract_faces(
                            frame, detector_backend='fastmtcnn')
                        # show a rectangle for each face
                    except Exception as e:
                        self.__results = None

                if self.__results is not None:
                    for result in self.__results:
                        x = result["facial_area"]["x"]
                        y = result["facial_area"]["y"]
                        w = result["facial_area"]["w"]
                        h = result["facial_area"]["h"]

                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)

                cv2.imshow('Frame', frame)
                self.__frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()

        except Exception as e:
            print("Error: {}.".format(e))


if __name__ == "__main__":
    cam = Camera('config.json', isTest=True)
    cam.start()
