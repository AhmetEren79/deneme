import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6,7])
a=["turkey","usa","amerika","rusya","diger","d","cs"]
y= x*2+5

plt.bar(a,y)
plt.xlabel("a")
plt.ylabel("y")
plt.show()