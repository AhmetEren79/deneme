import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

x_l = np.load("X.npy")
Y_l = np.load("Y.npy")
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')


# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
print("********************************")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)
print("********************************")
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# --------------LOGİSTİG REGRESSİON-------------------

# def initialize_weights_and_bias(dimension):
#     w = np.full((dimension,1),0.01)
#     b = 0.0
#     return w, b
#
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
#
# def forward_propagation(w,b,x_train,y_train):
#     z = np.dot(w.T,x_train) + b
#     y_head = sigmoid(z) # probabilistic 0-1
#     loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
#     cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
#     return cost
#
# def forward_backward_propagation(w,b,x_train,y_train):
#     # forward propagation
#     z = np.dot(w.T,x_train) + b
#     y_head = sigmoid(z)
#     loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
#     cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
#     # backward propagation
#     derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
#     derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
#     gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
#     return cost,gradients
#
# def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
#     cost_list = []
#     cost_list2 = []
#     index = []
#     # updating(learning) parameters is number_of_iterarion times
#     for i in range(number_of_iterarion):
#         # make forward and backward propagation and find cost and gradients
#         cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
#         cost_list.append(cost)
#         # lets update
#         w = w - learning_rate * gradients["derivative_weight"]
#         b = b - learning_rate * gradients["derivative_bias"]
#         if i % 10 == 0:
#             cost_list2.append(cost)
#             index.append(i)
#             print ("Cost after iteration %i: %f" %(i, cost))
#     # we update(learn) parameters weights and bias
#     parameters = {"weight": w,"bias": b}
#     plt.plot(index,cost_list2)
#     plt.xticks(index,rotation='vertical')
#     plt.xlabel("Number of Iterarion")
#     plt.ylabel("Cost")
#     plt.show()
#     return parameters, gradients, cost_list
# #parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
#
# def predict(w,b,x_test):
#     # x_test is a input for forward propagation
#     z = sigmoid(np.dot(w.T,x_test)+b)
#     Y_prediction = np.zeros((1,x_test.shape[1]))
#     # if z is bigger than 0.5, our prediction is sign one (y_head=1),
#     # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
#     for i in range(z.shape[1]):
#         if z[0,i]<= 0.5:
#             Y_prediction[0,i] = 0
#         else:
#             Y_prediction[0,i] = 1
#
#     return Y_prediction
#
#
# def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
#     # initialize
#     dimension = x_train.shape[0]  # that is 4096
#     w, b = initialize_weights_and_bias(dimension)
#     # do not change learning rate
#     parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
#
#     y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
#     y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
#
#     # Print train/test Errors
#     print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
#
#
# logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=150)

# ---------------------Artificial Neural Network (ANN)--------------------------
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters


def forward_propagation_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost_NN(A2, Y, parameters=None):
    eps = 1e-8  # sayısal stabilite
    cost = -np.mean(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))
    return cost


# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


# update parameters
def update_parameters_NN(parameters, grads, learning_rate=0.01):
    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
                  "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
                  "bias2": parameters["bias2"] - learning_rate * grads["dbias2"]}

    return parameters

# prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    # initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
        # forward propagation
        A2, cache = forward_propagation_NN(x_train, parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
        # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
        # update parameters
        parameters = update_parameters_NN(parameters, grads)

        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    # predict
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters


parameters = two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations=2500)


