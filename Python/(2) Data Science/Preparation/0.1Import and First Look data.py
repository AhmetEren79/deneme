import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../pokemon.csv")

# # Sadece sayısal sütunları seç
# numeric_data = data.select_dtypes(include=[np.number])
# # Heatmap çiz
# f, ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(numeric_data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()


# print(data.head())

# --------------------------------   LİNE PLOT--------------------------------------------------
# data.Speed.plot(kind = 'line',color ="g",label = "Speed",linewidth = 1,alpha=0.5,grid = True)
# data.Defense.plot(kind = 'line',color = "r",label = "Defense",linewidth =1,alpha=0.5,grid=True)
# plt.legend()
# plt.xlabel("x axis")
# plt.ylabel("y axis")
# plt.title("Line Plot")
# plt.show()

# --------------------------------   SCATTER PLOT--------------------------------------------------
# plt.scatter(data.Attack,data.Defense,color="red",alpha=0.5)
# plt.show()


# --------------------------------   HİSTOGRAM PLOT--------------------------------------------------
plt.hist(data.Speed,bins=50)
plt.show()