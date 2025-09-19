import numpy as np

array = np.array([1,2,3,4,5,6,7])

print(array[0])
print(array[0:4])  # 0'DAN 4.İNDEKSE KADAR GİDER

reserve_array= array[::-1]   # TERSİNİ ALIR
print(reserve_array)


array2= np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(array2[1,1])
print(array2[:,1])

print(array2[1,1:4])
print(array2[-1,:])      # EN SONDAKİ SATIRIN HEPSİNİ ALIR
print(array2[:,-1])      # ENN SONDAKİ SUTÜNÜN HEPSİNİ ALIR

print("***********************************************")

array3 = np.array([[1,2,3],[4,5,6],[7,8,9]])

a=array3.ravel()           #   SANKİ TEK SATIRMIŞ GİBİ YAPAR    [1,2,3,4,5,6,7,8,9]
print(a)
b = a.reshape(3,3)    # 3'E 3'LÜK BİR MATRİS YAPAR   * ORİJİNALİ DEĞİŞTİRMEZ BİR YERE ATAMAK LAZIM
arrayT=array3.T
print(b)
print(arrayT)              # TRANSPOZU

array4 = np.array([[1,2,3],[4,5,6],[7,8,9]])
array4.resize(5,2)         #  DİREK ORİJİNALİ DEĞİŞTİRİR RESHAPE'DEN FARKI BU   REFERANSLANDIYSA KABUL ETMEZ
print(array4)
