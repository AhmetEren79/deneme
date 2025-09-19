import numpy as np
import pandas as pd
dictionary = {"Name":["Ali","Veli","Mehmet","Ahmet","Mert","Kemal"],
              "Age":[32,40,50,22,10,38],
              "MAAS":[1000,2500,3750,1750,6215,3214]}

dataFrame1=pd.DataFrame(dictionary)
dataFrame1["Yeni Feature"] = [-1,-2,-3,-4,-5,-6]
ortalama= np.mean(dataFrame1.MAAS)

dataFrame1["MAAS_seviyesi"] = ["Yuksek" if(ortalama<each) else "Dusuk" for each in dataFrame1.MAAS]
dataFrame1.columns = [each.split()[0]+"_"+each.split()[1]  if(len(each.split())>1) else each for each in dataFrame1.columns]
dataFrame1.columns = [each.lower()  for each in dataFrame1.columns]

dataFrame1.drop(["yeni_feature"],axis=1,inplace=True)  # Yeni feature sutününü siler ve orijinale eşitler inplace sayesinde

dataFrame1["list_comp"] = [each*2 for each in dataFrame1.age]

def multiply(age):
    return age*2

dataFrame1["deneme2"] = dataFrame1.age.apply(multiply)

print(dataFrame1)