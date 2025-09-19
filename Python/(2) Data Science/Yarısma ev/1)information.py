import numpy as np
import matplotlib as plt
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.columns)
print("********************")
print(train_df.head())
print("********************")
print(train_df.describe())
print("********************")
print(train_df.info())