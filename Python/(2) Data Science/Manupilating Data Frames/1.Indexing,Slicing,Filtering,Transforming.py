from operator import index

import pandas as pd

data = pd.read_csv("../pokemon.csv")
data = data.set_index("#")

# print(data.loc[2,["HP"]])

data2=data.head()
# print(data2[["HP","Attack"]])
# print(data2.loc[1:3,"HP":"Defense"])      # HP DEN DEFENSE'YE KADAR
# print(data2.loc[1:3,["HP","Defense"]])    # SADECE HP VE DEFENSE
# print(data2.loc[3:1:-1,"HP":"Defense"])   # 3 2 1 diye yazdırır
# print(data2.loc[1:3,"Speed": ])             # Speed den sonrakileri alır

# az_Can=data2["HP"]<100
# print(az_Can)
# print(data2[az_Can])
print("*********************")
# ---------------------FİLTER---------------------
first_filter = data["HP"]<100
second_filter=data["Defense"]>150
# print(data[first_filter & second_filter])

deneme=data[first_filter & second_filter]
print(deneme.loc[:,["HP","Defense"]])
print("*********************")

print(data.HP[data.Speed<15])         # Böyle de Gösterilebilir
print("------------------------------------------------")
# ------------------------------------------
data["total_power"]=data.Attack+data.Defense
print(data.head())
print("------------------------------------------------")
# ------------------------------------------
data2.index.name ="index_name"
print(data.head())
print("------------------------------------------------")
# ------------------------------------------
data3=data.copy()
data3.index=range(100,900,1)
print(data3.head())
print("------------------------------------------------")
# ------------------------------------------
data4=data3.set_index(["Type 1","Type 2"])
print(data4.index)
# print(data4)
print("_________________________________________________________")

# ---------------------Pivoting---------------------

dic={"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
dt=pd.DataFrame(dic)
print(dt)
print(dt.pivot(index="treatment",columns="gender",values="response"))
print("___________________")
dt1=dt.set_index(["treatment","gender"]) #****************************************
print(dt1)

print(dt1.unstack(level=0))
print("___________________")
print(dt1.unstack(level=1))
print("___________________")

# ---------------------Melting---------------------

dt_melted = pd.melt(dt,id_vars="treatment",value_vars=["age","response"])
print(dt_melted)

# ------------------------------------------
print("___________________")
dt2=dt.copy()
print(dt2.groupby("treatment").age.mean())    # min,max ,sum vbvb
print("___________________")
print(dt2.groupby("treatment")[["age","response"]].min())
