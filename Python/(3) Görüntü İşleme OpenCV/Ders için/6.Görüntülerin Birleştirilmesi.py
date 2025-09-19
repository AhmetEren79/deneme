import cv2
import numpy as np

img =cv2.imread("../images/Lenna.png")

# Yatay (Horizontal)
hor= np.hstack((img,img))
# Dikey (Vertical)
ver = np.vstack((img,img))

cv2.imshow("Original",img)
cv2.imshow("Horizontal",hor)
cv2.imshow("Vertical",ver)
cv2.waitKey()
cv2.destroyAllWindows()
