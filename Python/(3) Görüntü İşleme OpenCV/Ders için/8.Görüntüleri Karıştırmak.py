import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("../images/img1.JPG")
img2 = cv2.imread("../images/img2.JPG")

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.imshow(img1)
#
# plt.figure()
# plt.imshow(img2)

print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1,(600,600))
img2 = cv2.resize(img2,(600,600))

# plt.figure()
# plt.imshow(img1)
# plt.figure()
# plt.imshow(img2)
# plt.show()

# Karıştırılmış Resim =alpha*img1+beta*img2
blended = cv2.addWeighted(src1=img1,alpha=0.4,src2=img2,beta=0.8,gamma=0)
plt.figure()
plt.imshow(blended)
plt.show()