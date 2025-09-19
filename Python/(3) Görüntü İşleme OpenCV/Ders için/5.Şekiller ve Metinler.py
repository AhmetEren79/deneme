import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
print(img.shape)

# Çizgi
# (Resim,Başlangıç Yeri,Bitiş Yeri, Renk, Kalınlık )
img_line = cv2.line(img,(100,100),(100,300),(0,255,0),3)

# Dikdörtgen
# (Resim,Başlangıç Yeri,Bitiş Yeri, Renk, Kalınlık )
img_dik= cv2.rectangle(img,(0,0),(256,256),(255,0,0),cv2.FILLED)

# Cember
# (Resim,Merkez,Yarıçap,renk)
img_cem = cv2.circle(img,(300,300),45,(0,0,255),cv2.FILLED)

# Metin
cv2.putText(img,"Resim",(350,350),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))

cv2.imshow("Original",img)
cv2.imshow("Cizgi",img_line)
cv2.imshow("DİK",img_dik)
cv2.imshow("Cember",img_cem)
cv2.imshow("or2",img)
cv2.waitKey()
cv2.destroyAllWindows()