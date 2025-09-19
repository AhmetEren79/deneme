from collections import Counter
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


def detect_outlier(df,features):
    outlier_indices=[]

    for each in features:
        Q1 = np.percentile(df[each], 25)
        Q3 = np.percentile(df[each],75)
        IQR = Q3-Q1

        outlier_step = IQR*1.5

        outlier_list_col = df[(df[each] < Q1 - outlier_step) | (df[each]>Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)

    return multiple_outliers

print(train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])])

train_df = train_df.drop(detect_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

