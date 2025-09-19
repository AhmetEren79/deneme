import numpy as np
import pandas as pd

data = pd.read_csv("../pokemon.csv")

data_new= data["Type 2"].value_counts(dropna = False)   # dropna nan olanları da gösterir
print(data_new)
print(data.loc[:10,["Name","Type 1","Type 2"]])
print("***********************")
data1 = data.copy()
data1["Type 2"].dropna(inplace=True)       #?????????????????????
print(data1.loc[:10,["Name","Type 1","Type 2"]])

# --------------------- ASSERT---------------------
assert 1==1
assert data1["Type 2"].notnull().all()
assert data.Speed.dtypes == np.int64
