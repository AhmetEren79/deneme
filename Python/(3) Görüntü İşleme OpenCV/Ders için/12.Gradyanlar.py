import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread("../images/sudoku.jpg",0)

plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.title("Original")

#    x Gradyan

sobelx = cv2.Sobel(img,ddepth=cv2.CV_16S, dx=1,dy=0,ksize=5)
plt.figure()
plt.imshow(sobelx,cmap="gray")
plt.axis("off")
plt.title("X Gradyan")

#  y Gradyan
sobely = cv2.Sobel(img,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=5)
plt.figure()
plt.imshow(sobely,cmap="gray")
plt.axis("off")
plt.title("Y Gradyan")
plt.show()

