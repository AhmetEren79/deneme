import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv("iris.csv")

setosa = df[df.Species == "Iris-setosa"]
versicolor =df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]


plt.subplot(2,1,1)
plt.plot(setosa.Id,setosa.PetalLengthCm, color = "red",label="setosa")
plt.subplot(2,1,2)
plt.plot(versicolor.Id,versicolor.PetalLengthCm,color = "green",label="versicolor")

plt.show()