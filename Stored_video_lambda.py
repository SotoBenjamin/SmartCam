import json
import boto3
import cv2
import math


def analyzeVideo():
    videoFile = "cosa.mp4"

  #  rekognition = boto3.client('rekognition')
    ppeLabels = []
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)  # frame rate
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        print("Processing frame id: {}".format(frameId))
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            hasFrame, imageBytes = cv2.imencode(".jpg", frame)


   #        if (hasFrame):
    #            response = rekognition.detect_protective_equipment(
     #               Image={
      #                  'Bytes': imageBytes.tobytes(),
       #             }
        #        )

         #   for person in response["Persons"]:
          #      person["Timestamp"] = (frameId / frameRate) * 1000
           #     ppeLabels.append(person)
    cv2.imshow('frame', frame)
    print(ppeLabels)

  # with open(videoFile + ".json", "w") as f:
   #     f.write(json.dumps(ppeLabels))

    cap.release()


if __name__ == "__main__":
    analyzeVideo()
