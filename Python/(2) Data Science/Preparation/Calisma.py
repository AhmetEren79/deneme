import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)

data = pd.read_csv("../pokemon.csv")

print(data.columns)
print("******************")
print(data.info)
print("******************")
print(data.describe())
print("******************")


x = data["HP"]==1
print(data[x])
print("******************")
data_legend=data["Legendary"] == True
print(data[data_legend])
# print("******************")
#  y = data[(data["Attack"]>180) & (data["Defense"]<110)]
#  print(y)


#  ----------------------------- LİNE -----------------------------------------
# data.Attack.plot(kind = "line",color="red",label="Attack",alpha=1,grid=True)
# data.Defense.plot(kind="line",color="green",label="Defence",grid=True,alpha=1)
# plt.legend()
# plt.show()
#  ----------------------------- Histogram -----------------------------------------
# plt.hist(data.Defense,bins=100)
# plt.show()
#  ----------------------------- Scatter -----------------------------------------
# plt.scatter(data.Attack,data.Defense,color="green")
# plt.show()
