import pandas as pd

dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1=pd.DataFrame(dictionary)

print(dataFrame1["Name"])     # SADECE NAME'İ ALIR
print("*****************")
print(dataFrame1.Name)        # AYNISI ANCAK FARKLI GÖSTERİMİ
print("*****************")

dataFrame1["Yeni_Feature"] = [-1,-2,-3,-4,-5,-6]   # YENİ SUTÜN EKLEME
print(dataFrame1)
print("*****************")
print(dataFrame1.loc[:,"Age"])    # AGE'DEKİ HEPSİNİ GÖSTERİR
print("*****************")
print(dataFrame1.loc[:3,"Age"])   # İLK 3 SATIRINI GÖSTERİR   (3 de dahil olur)  Nuumpy da sayılmıyor.
print("*****************")
print(dataFrame1.loc[:,"Name":"MAAS"])   # NAME'DEN AGE'YE KADAR YAZDIR
print("*****************")
print(dataFrame1.loc[:,["Age","Name"]])  # SADECE BU İKİSİNİ YAZDIRIR
print("*****************")
print(dataFrame1.loc[::-1,:])      # TERSİNİ VERİR
print("*****************")
print(dataFrame1.loc[:,:"Age"])   # AGE'E KADAR OLANLARI YAZDIRIR
print("*****************")
print(dataFrame1.iloc[:,2])    # O İNDEKSTEKİNİ YAZDIRIR