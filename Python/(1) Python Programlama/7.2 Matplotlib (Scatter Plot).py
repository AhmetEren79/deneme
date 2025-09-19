import  matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("iris.csv")

setosa = df[df.Species == "Iris-setosa"]
versicolor =df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.scatter(setosa.Id,setosa.PetalLengthCm, color = "red",label="setosa")
plt.scatter(versicolor.Id,versicolor.PetalLengthCm,color = "green",label="versicolor")
plt.scatter(virginica.Id,virginica.PetalLengthCm,color="blue",label="virginica")
plt.legend()
plt.xlabel("id")
plt.ylabel("PetalLengthCm")
plt.title("aaaaaaa")
plt.show()


