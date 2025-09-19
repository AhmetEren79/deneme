
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("excelller/Excel1.csv",sep = ";")

y = df.hiz.values.reshape(-1,1)
x = df.fiyat.values.reshape(-1,1)

lr = LinearRegression()

lr.fit(x,y)

y_head = lr.predict(x)

print("10 milyon tl lik araba hizi tahmini: ",lr.predict([[12]]))

polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.scatter(x,y)
plt.plot(x,y_head2,color= "green",label = "poly")
plt.plot(x,y_head,color="red",label ="linear")
plt.legend()
plt.show()