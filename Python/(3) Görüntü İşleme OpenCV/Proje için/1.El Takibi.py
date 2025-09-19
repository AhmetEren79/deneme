import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(imgRGB)
    fingersUp = 0
    message = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            lmList = []
            h, w, c = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if lmList:
                # Başparmak kontrolü (x ekseni)
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingersUp += 1

                # Diğer 4 parmak (y ekseni)
                for i in range(1, 5):
                    if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]:
                        fingersUp += 1

    # FPS hesapla
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Mesaj belirleme
    if fingersUp == 1:
        message = "1 Parmak Yukari!"
    elif fingersUp == 2:
        message = "Zafer!"
    elif fingersUp == 5:
        message = "Selam!"

    # FPS yaz
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)

    # Parmak sayısı yaz
    cv2.putText(img, f"Parmak Sayisi: {fingersUp}", (10, 150),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    # Mesaj yaz (varsa)q
    if message:
        cv2.putText(img, message, (10, 230),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
