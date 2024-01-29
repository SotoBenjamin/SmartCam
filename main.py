import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # ret if it works propiertly // frame the actuall frame
    width = int(cap.get(3))  # width of frame
    height = int(cap.get(4))  # height of frame

    #image = np.zeros(frame.shape, np.uint8)
#    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow()
