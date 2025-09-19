import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../images/kart.png")
print(img.shape)

width=400
height=500

pts1 = np.float32([[230,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)          #perspektif 4 nokta alır affine 3 nokta alır ***************
print(matrix)

img_Output = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Original",img)
cv2.imshow("Nihai",img_Output)
cv2.waitKey()
cv2.destroyAllWindows()


# plt.imshow(img)
# plt.show()


