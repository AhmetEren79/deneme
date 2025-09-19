import numpy as np

a=np.array([1,2,3])
b=np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)
print("*************************************************")
print(np.sin(a))    # a'daki değerlerin sinüsü
print(a<2)          # a'daki değerler 2 den lüçük mü döndürür boolean olarak

print("*************************************************")
print(a*b)
print(a.dot(b))                # Matrix çarpımı ?????????
print("*************************************************")
print(np.exp(a))               # exponansiyel değeri
print("*************************************************")
c=np.random.random((3,3))

print(c)
print(c.sum())
print(c.sum(axis=0))   #   SÜTUNLARI TOPLAR
print(c.sum(axis=1))   #   SATIRLARI TOPLAR
print(c.max())
print(c.min())

print("*************************************************")
print(a)
print(np.sqrt(a))        # KAREKÖKÜNÜ ALDIK
print(np.square(a))      # KARESİNİ ALDIK
print(np.add(a,a))       # A'YI A'YA EKLER TOPLAR
