import cv2
import pickle
import numpy as np

def checkParkSpace(imgg):
    spaceCounter = 0

    for pos in posList:
        x,y = pos
        img_crop = imgg[y:y+height,x:x+width]
        count = cv2.countNonZero(img_crop)

        if count<150:
            color = (0,255,0)
            thickness = 2
            spaceCounter +=1
        else:
            color = (0,0,255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

width = 27
height = 15

cap = cv2.VideoCapture("../ProjeResimvevideo/video.mp4")

with open("CarParkPose","rb") as f:
    posList = pickle.load(f)

while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian = cv2.medianBlur(imgThreshold,5)
    imgDilate = cv2.dilate(imgMedian,np.ones((3,3)),iterations=1)

    checkParkSpace(imgDilate)

    cv2.imshow("img", img)
    # cv2.imshow("imgGray",imgGray)
    # cv2.imshow("imgblur",imgBlur)
    # cv2.imshow("imgthreshold",imgThreshold)
    # cv2.imshow("imgMedian",imgMedian)
    # cv2.imshow("imgDilate",imgDilate)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
