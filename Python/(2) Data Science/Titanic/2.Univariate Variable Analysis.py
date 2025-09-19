import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import category
sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#-------------------- Categorical Variable --------------------   2 veya daha fazla kategori(Seçenek) varsa buna girer
# Survived,Sex,Pclass,Embarked,Cabin,Name,Ticket,Sibsp and Parch

def ba_plot(variable):
    """
       input: variable ex: "Gender"
       output:bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variables(value/simple)
    var_value = var.value_counts()
    # visualize
    plt.figure(figsize=(9,3))
    plt.bar(var_value.index,var_value)
    plt.xticks(var_value.index,var_value.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,var_value))

def hist_plot(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


category1 =["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

# for each in category1:
#     ba_plot(each)
#     print("**********")

category2=["Cabin","Name","Ticket"]

# for each in category2:
#     print("{} \n ".format(train_df[each].value_counts()))
#     print("**********")

#-------------------- Numerical Variable --------------------

numeric_Var=["Fare","Age","PassengerId"]
for each in numeric_Var:
    hist_plot(each)

# abc=train_df["Fare"]>450
# print(train_df[abc])