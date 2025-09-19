import numpy as np

array= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])  # 1*15 VEKTÖR
print("Array :",array)
print("Array shape is :",array.shape)   # BOYUTU KAÇA .shape
print("Array Dimension is :",array.ndim)

print("*************************************************")

a=array.reshape(3,5)   #   3*5 VEKTÖR HALİNE GETİRİYORUZ
print("3*5 Vektör oldu:\n",a)
print("Shape is :",a.shape)
print("Dimention is :",a.ndim)    # KAÇ BOYUTLU .ndim
print("Data Type is :",a.dtype.name)   # DATA TYPE İSMİ

print("*************************************************")
zeros = np.zeros((3,4))           # 0 LARDAN OLUŞAN MATRİX  3'E 4
zeros[0][0] = 5
print(zeros)

print("*************************************************")
print(np.ones((3,4)))             # 1 LERDEN OLUŞAN MATR1İX 3'E 4

print("*************************************************")
print(np.empty((2,3)))            # İÇİ BOŞ OLAN  2 YE 3'LÜK MATRİX

print("*************************************************")
b=np.arange(10,50,5)              #    10'DAN 50'YE KADAR (DAHİL DEĞİL ) 5'ER 5'ER GİDER
print(b)

print("*************************************************")
c=np.linspace(10,50,5)   # 10'DAN 50'YE KADAR (DAHİL) 5 SAYI OLACAK ŞEKİLDE YAZAR
print(c)

