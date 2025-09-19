import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../pokemon.csv")

data.boxplot(column="Attack",by="Legendary")
plt.show()

deneme =data.loc[:5,["Name","Attack","Defense"]]
print(deneme)