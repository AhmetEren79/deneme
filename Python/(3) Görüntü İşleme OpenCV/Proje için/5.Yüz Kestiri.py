import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.20)

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result =faceDetection.process(imgRGB)

    if result.detections:
        for id,detection in enumerate(result.detections):
            bboxC = detection.location_data.relative_bounding_box
            h,w,_ = img.shape
            bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img,bbox,(255,0,0),2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break