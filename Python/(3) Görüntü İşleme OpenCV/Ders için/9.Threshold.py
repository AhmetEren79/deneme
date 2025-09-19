import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/img1.JPG",0)


_,thresh_img = cv2.threshold(img,thresh=60,maxval=255,type= cv2.THRESH_BINARY)
thresh2_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)

cv2.imshow("img",img)
cv2.imshow("thresh1",thresh_img)
cv2.imshow("thresh2",thresh2_img)
cv2.waitKey()
cv2.destroyAllWindows()