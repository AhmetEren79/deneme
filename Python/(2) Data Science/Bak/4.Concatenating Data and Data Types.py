import pandas as pd

data =pd.read_csv("../pokemon.csv")

data1= data.head()
data2 = data.tail()

data_new = pd.concat([data1,data2],axis=0,ignore_index=True)
print(data_new)
print("****************")

data3=data["Attack"].head()
data4 = data["Defense"].head()

data2_new = pd.concat([data3,data4],axis=1)
print(data2_new)
# --------------------- ---------------------
print(data.dtypes)
data["Type 1"] = data["Type 1"].astype("category")    # *********************************
data["Speed"] = data["Speed"].astype("float")         # Birbirine çevirme çok kullanılır..
print(data.dtypes)
