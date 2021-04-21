import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

### Part 1

d = 100 # dimensions of data
train_n = 100 # number of data points

X = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
a_zeroes = np.random.normal(0,0, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1)) # approximation plus some noise
y_zeroes = X.dot(a_zeroes) # zeroes with no noise

def solve_exact(X,y):
    a_hat = np.linalg.inv(X).dot(y)
    return a_hat

# 1.a
def l2_reg(X, y):
    X_T = X.transpose()
    XTX = np.matmul(X,X_T)
    XTX_inv = np.linalg.inv(XTX)
    a_hat = np.matmul(X_T,XTX_inv).dot(y)
    return a_hat

a_hat = l2_reg(X, y)

y_hat = X.dot(a_hat) + np.random.normal(0,0.5,size=(train_n,1))
y_zeroes = X.dot(a_zeroes) + np.random.normal(0,0.5,size=(train_n,1))

print(y_hat, y_zeroes)

# 1.b gradient descent
def get_gradient(X,a,y):
    return (2*(X.dot(a)-y)*X).reshape(-1,1)

def descend_gradient(a_guess, X, y, step_size, n_train, n_iter=1000):
    a_hat = a_guess
    for i in range(n_iter):

        x_i = X[i]
        y_i = y[i]
        a_hat = a_hat - (step_size * get_gradient(x_i, a_hat, y_i))
    return a_hat


step_sizes = [0.00005, 0.0005, 0.0007]

# 1.c stochastic gradient descent using ben's method
train_n = 100

def error(X,a,y):
    return np.square(np.linalg.norm(X.dot(a)-y))

def sgd(a_guess, X, y, alpha, n=train_n, n_iter=1000):
    a_hat = a_guess
    iter_number = []
    obj_fun_values = []

    for i in range(n_iter):
        idx = np.random.randint(n)
        x_i = X[idx]
        y_i = y[idx]
        grad = get_gradient(x_i, a_hat, y_i)
        a_hat = a_hat - alpha * grad
        iter_number += [i]
        obj_fun_values += [error(x_i, a_hat, y_i)]

    return a_hat, iter_number, obj_fun_values

step_sizes = [0.00005, 0.005, 0.01]
sgd_preds = []

a_hat_sgd_01, iter_numbers_01, obj_fun_values_01 = sgd(a_hat, X, y, 0.005, 100, 1000)
a_hat_sgd_005, iter_numbers_005, obj_fun_values_005 = sgd(a_hat, X, y, 0.005, 100, 1000)
a_hat_sgd_0005, iter_numbers_0005, obj_fun_values_0005 = sgd(a_hat, X, y, 0.0005, 100, 1000)

plt.plot(iter_numbers_0005, obj_fun_values_0005, 'green')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (Error)')
plt.savefig('figs/part1c.sgd_step0005.png')
plt.clf()

plt.plot(iter_numbers_005, obj_fun_values_005, 'blue')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (Error)')
plt.savefig('figs/part1c.sgd_step005.png')
plt.clf()

plt.plot(iter_numbers_01, obj_fun_values_01, 'orange')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (Error)')
plt.savefig('figs/part1c.sgd_step01.png')
plt.clf()
