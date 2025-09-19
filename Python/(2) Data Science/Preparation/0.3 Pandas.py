import pandas as pd
pd.set_option("display.max_columns", None)
data = pd.read_csv("../pokemon.csv")

# series = data["Defense"]   # SERİES   vektör şeklinde
dataFrame =data[["Defense"]]   # DATA FRAMES  2 boyutlu

x = data["Defense"] >200
print(data[x])
print("*******************")
y=data[(data["Defense"]>200) & (data["Attack"]>100)]
print(y)
print("*******************")
for index,value in data[["Attack"]][0:1].iterrows():
    print(index,":",value)