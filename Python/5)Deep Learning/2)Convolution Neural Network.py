import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
print(train.shape)
print(train.head())
print("************************************")
test= pd.read_csv("test.csv")
print(test.shape)
print(test.head())

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)


# visualize number of digits classes
# plt.figure(figsize=(15,7))
# g = sns.countplot(x=Y_train, hue=Y_train,palette="icefire")
# plt.title("Number of digit classes")
# Y_train.value_counts()
# plt.show()

# img = X_train.iloc[0].to_numpy()
# img = img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.title(train.iloc[0,0])
# plt.axis("off")
# plt.show()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

from keras.utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)