import pandas as pd
# --------------------- Tidy Data---------------------
data =pd.read_csv("../pokemon.csv")

new_data= data.head()

melted=pd.melt(frame=new_data,id_vars="Name",value_vars=["Attack","Defense"])
print(melted)

# --------------------- Pivot Data--------------------- Eski haline getirme
un_melted=melted.pivot(index="Name",columns="variable",values="value")
print(un_melted)

