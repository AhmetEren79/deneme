import cv2

img = cv2.imread("../images/Lenna.png")
print("Resim boyutu: ",img.shape)

# resize
img_Resized = cv2.resize(img,(600,600))
# kırp
img_Cropped = img[:200,0:300]

cv2.imshow("Original",img)
cv2.imshow("Resized",img_Resized)
cv2.imshow("Kirpik resim",img_Cropped)
cv2.waitKey()
cv2.destroyAllWindows()


