import pandas as pd

dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1 = pd.DataFrame(dictionary)

print(dataFrame1)

print("***************************************")
head = dataFrame1.head()    # EN ÜSTTEN 5 TANESİNİ ALIR
head2 = dataFrame1.head(3)   # EN ÜSTTEN 3 TANESİNİ ALIR
print(head)
print("***************************************")
tail = dataFrame1.tail()     # EN SONDAKİ 5 TANESİ
tail2 = dataFrame1.tail(3)   # EN SONDAN 3 TANESİ
print(tail2)

