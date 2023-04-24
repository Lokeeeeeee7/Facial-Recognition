import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from PIL import ImageGrab
import mediapipe as mp
import time

                #face recognition
pTime = 0
path = "D:\#BK\XLA\BTL\Python BTL\ImageAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


# markAttendance('Elon')

# def captureScreen(bbox=(0, 0, 1550, 800)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr



encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("person 1.mp4")'
            #face recognition
            # Face landmarks
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
            # Face landmarks
#face recognition
while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
# face recognition
    # face landmarks
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # face landmarks
# face recognition
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
# face recognition
    # face landmarks
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
    # face landmarks
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        # if matches[matchIndex]:
        #     name = classNames[matchIndex].upper()
        #     print(name)
        #     y1, x2, y2, x1 = faceLoc
        #     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #     cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #     markAttendance(name)
        if faceDis[matchIndex]:
            name = classNames[matchIndex].upper()
            # name = 'Unknown'
        else:
            name = 'Unknown'
        markAttendance(name)
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('webcam', img)
    cv2.waitKey(200)

cap.release()
cv2.destroyAllWindows()
