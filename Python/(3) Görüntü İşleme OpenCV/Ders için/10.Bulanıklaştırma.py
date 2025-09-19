import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../images/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# plt.figure()
# plt.imshow(img)
# plt.axis("off")
# plt.title("Orijinal")


"""
   ORTALAMA BULANIKLAŞTIRMA
"""

# dst2 = cv2.blur(img,ksize=(3,3))
# plt.figure()
# plt.imshow(dst2)
# plt.axis("off")
# plt.title("Ortalama Blur")


"""
   Gaussion Bulanıklaştırma
"""

# gb = cv2.GaussianBlur(img,ksize=(3,3),sigmaX=7)
# plt.figure()
# plt.imshow(gb)
# plt.axis("off")
# plt.title("Gaussion Blur")


"""
   Medyan Bulanıklaştırma
"""
# mb = cv2.medianBlur(img,ksize=3)
# plt.figure()
# plt.imshow(mb)
# plt.axis("off")
# plt.title("Medyan Blur")
# plt.show()d


def gaussionNoise(image):
    row,col,ch = image.shape
    mean=0
    var = 0.05
    sigma = var**0.5

    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy=image+gauss
    return noisy

#  İçe Aktar ve Normalize Et

img2 = cv2.imread("../images/NYC.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Orijinal")

# gaussion_noise = gaussionNoise(img2)
# plt.figure()
# plt.imshow(gaussion_noise)
# plt.axis("off")
# plt.title("gaussion noisy")


#  Gaussion Blur
# gb2 = cv2.GaussianBlur(img2,ksize=(3,3),sigmaX=7)
# plt.figure()
# plt.imshow(gb2)
# plt.axis("off")
# plt.title("Gaussion blur")

def saltPepperNoise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(image)

    # Salt Beyaz
    num_salt=np.ceil(amount*image.size*s_vs_p)
    coords= [np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    # Pepper Siyah
    num_pepper = np.ceil(amount * image.size * (1-s_vs_p))
    coords2 = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords2)] = 0

    return noisy

spImage=saltPepperNoise(img)
plt.figure()
plt.imshow(spImage)
plt.axis("off")
plt.title("Salt and Pepper")

mb2 = cv2.medianBlur(spImage.astype(np.uint8),ksize=3)
plt.figure()
plt.imshow(mb2)
plt.axis("off")
plt.title("Median Blur")
plt.show()