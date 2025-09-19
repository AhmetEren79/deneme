import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# --------------- Find the Missing Data-----------------

train_df_len=len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)   # *******

print(train_df.head())
print("**************")
print(train_df.columns[train_df.isnull().any()])    # Hangi features de null var bakar
print("**************")
print(train_df.isnull().sum())    # kaç null var bakar
print("**************")

# --------------- Fİll the Missing Data-----------------

print(train_df[train_df["Embarked"].isnull()])                # Boş olan Embarklar neler bakar

# train_df.boxplot(column="Fare",by="Embarked")
# plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")        # Boş Embarkedi  C ile değiştirdik
print("**************")
print(train_df[train_df["Embarked"].isnull()])
print("******************************************")    #   Fare için ise
print(train_df[train_df["Fare"].isnull()])

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))    #  Fare Doldurduk
print(train_df[train_df["Fare"].isnull()])