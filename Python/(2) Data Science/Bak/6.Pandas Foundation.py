import pandas as pd
# --------------------List den dict den data frame yapma
country=["Spain","Germany"]
population=[11,12]
list_label=["Country","Population"]
list_col=[country,population]

zipped = list(zip(list_label,list_col))
dat_dict =dict(zipped)
df=pd.DataFrame(dat_dict)
print(df)
print("*************")
df["Capital"] = ["Madrid","Berlin"]
print(df)
