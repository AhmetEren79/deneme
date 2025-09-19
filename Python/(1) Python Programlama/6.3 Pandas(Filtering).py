import pandas as pd

dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1=pd.DataFrame(dictionary)


filtre1=dataFrame1.Age<25
filtrelenmis_data= dataFrame1[filtre1]
print(filtrelenmis_data)
print("****************************")
filtre2=dataFrame1.MAAS<2000
print(dataFrame1[filtre1 & filtre2])    # İki filtreyi birleştiriyor
print("****************************")
print(dataFrame1.Age>45)              # BOOLEAN OLARAK BAKAR
print("****************************")
print(dataFrame1[dataFrame1.Age>45])
