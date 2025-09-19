import numpy as np
import pandas as pd

dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1=pd.DataFrame(dictionary)
dataFrame1["Yeni Feature"] = [-1,-2,-3,-4,-5,-6]
ortalama_maas= dataFrame1.MAAS.mean()
ortalama_maas2=np.mean(dataFrame1.MAAS)         # ORTALAMA MAAS BULUYORUZ İKİSİNDE DE

dataFrame1["MAAS_Seviyesi"] = ["dusuk" if ortalama_maas>each else "Yuksek" for each in dataFrame1.MAAS]   # EKLEYİP SEVİYELERİ AYARLADIK TABLOYA

print(ortalama_maas)
print(ortalama_maas2)
print("************************")

for each in dataFrame1.MAAS:
    if(ortalama_maas>each):
        print("Dusuk")
    else:
        print("Yuksek")
print("************************")
print(dataFrame1)
print("************************")

dataFrame1.columns = [each.lower()  for each in dataFrame1.columns]
dataFrame1.columns = [each.split()[0]+"_"+each.split()[1]  if(len(each.split())>1) else each for each in dataFrame1.columns]
print(dataFrame1)