import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)


train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_PassengerId=test_df["PassengerId"]

print(train_df.columns)
print("********************")
print(train_df.head())
print("********************")
print(train_df.describe())
print("********************")
print(train_df.info())