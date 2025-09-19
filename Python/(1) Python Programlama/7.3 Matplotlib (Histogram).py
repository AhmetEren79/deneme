import  matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("iris.csv")

setosa = df[df.Species == "Iris-setosa"]
versicolor =df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.hist(setosa.PetalLengthCm,bins=50)
plt.xlabel("a")
plt.ylabel("b")
plt.title("c")
plt.show()
