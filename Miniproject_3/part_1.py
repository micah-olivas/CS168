import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

### Part 1

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
a_true = np.random.normal(0,1, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1)) # approximation plus some noise

# 1.a
def l2_reg(X, y):
    X_T = X.transpose()
    XTX = np.matmul(X,X_T)
    XTX_inv = np.linalg.inv(XTX)

    a = np.matmul(X_T,XTX_inv).dot(y)

    return a

l2_reg(X, y)

# 1.b gradient descent
def descend_gradient(vector, step_size, n_iter):
    for _ in range(n_iter):
        diff = -step_size * np.gradient(vector)
        vector += diff
    return vector

descend_gradient()

# 1.c stochastic gradient descent
def stochastic_gradDesc(vector, step_size, n_iter):



### Part 2

train_n = 100
test_n = 1000
d = 100
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
