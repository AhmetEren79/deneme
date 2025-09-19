import numpy as np

array1 = np.array([[1,2],[3,4]])
array2 = np.array([[-1,-2],[-3,-4]])

array3 = np.vstack((array1,array2))    # DİKEY OLARAK EKLER
array4 = np.hstack((array1,array2))    # YATAY OALRAK EKLER

print(array3)
print("***************")
print(array4)
print("***************************************************************")
         #         CONVERT ETMEK VE COPY ARRAY

liste= [1,2,3,4]
array5 = np.array(liste)     # listeyi array yapma

liste2 = list(array5)        # arrayi liste haline getirmek


#ab=np.array([1,2,3])
#bc=ab                 BİRİ DEĞİŞİNCE HEPSİ DEĞİŞİR ÇÜNKÜ DEĞER OLARAK DEĞİL ALAN OLARAK TUTULUR
#cd=ab                     BUNU YAPMAK YERİNE AŞAĞIDAKİ GİBİ YAPILMASI GEREKİR

x = np.array([1,2,3])
y=x.copy()
z=x.copy()
