
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("excelller/decisiontreeregressiondataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


tree_reg = DecisionTreeRegressor()   # random sate = 0
tree_reg.fit(x,y)


tree_reg.predict([[5.5]])
x_n = np.arange(x.min(), x.max(), 0.01).reshape(-1,1)
y_head = tree_reg.predict(x_n)

plt.scatter(x,y,color="red")
plt.plot(x_n,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()