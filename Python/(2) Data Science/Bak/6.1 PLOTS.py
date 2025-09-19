import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../pokemon.csv")
# ----------------------SUBPLOTS
data.plot(subplots =True)
plt.show()

# ----------------------SCATTER PLOTS
plt.scatter(data.Speed,data.Defense)
plt.xlabel("Speed")
plt.ylabel("Defense")
plt.show()

# ----------------------HİSTOGRAM PLOTS
plt.hist(data.Attack,bins=50)
plt.xlabel("Attack Power")
plt.ylabel("Pokemon Count")
plt.show()
# plt.savefig("Deneme.png")
