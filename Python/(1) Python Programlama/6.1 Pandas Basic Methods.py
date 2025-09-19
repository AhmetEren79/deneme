import pandas as pd

dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1 = pd.DataFrame(dictionary)
print("****************************************")
print(dataFrame1.columns)        # SUTÜNDAKİLERİ VERİR
print("****************************************")
print(dataFrame1.info())         # BİLGİ VERİR
print("****************************************")
print(dataFrame1.dtypes)         # İNFO GİBİ AMA DAHA SADE DATA TÜRLERİNİ VERİR
print("****************************************")
print(dataFrame1.describe())     # NUMERİC OLARAK AÇIKLAR