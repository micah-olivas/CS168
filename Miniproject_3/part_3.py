import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np

## Part 3

def norm_error(X, y, a):
    return np.linalg.norm( X @ a - y) / np.linalg.norm(y)

def random_weights(X_train, y_train, X_test, y_test):
    a_est = np.random.normal(0, 1, size=(X_train.shape[1], 1))

    train_error = np.linalg.norm( X_train @ a_est - y_train) / np.linalg.norm(y_train)
    test_error = np.linalg.norm( X_test @ a_est - y_test ) / np.linalg.norm(y_test)
    train_error = norm_error(X_train, y_train, a_est)
    test_error = norm_error(X_test, y_test, a_est)
    return a_est, train_error, test_error
    
def PCA_compute_exact(X_train, y_train, X_test, y_test, K = 100):
    u, s, vh = np.linalg.svd(X_train, full_matrices = False)

    # reduce dimensions to the number of training samples
    X_prime = X_train @ vh.T[:,:K]

    # compute exact L2 solution of X in lower dimension, than project back to full D dimensions
    a_est = vh.T[:,:K] @ np.linalg.inv(X_prime.T @ X_prime) @ X_prime.T @ y_train

    # compute normalized testing error
    train_error = norm_error(X_train, y_train, a_est)
    test_error = norm_error(X_test, y_test, a_est)
    return a_est, train_error, test_error

def Elastic_net(X_train, y_train, X_test, y_test):
    regr = ElasticNet(fit_intercept=False, l1_ratio = .9, alpha = .001)
    regr.fit(X_train, y_train)

    # project the estimate from elastic net fit to full D dimensions
    a_est = regr.coef_

    # compute normalized testing error
    train_error = norm_error(X_train, y_train, a_est)
    test_error = norm_error(X_test, y_test, a_est)
    return a_est, train_error, test_error

def Lasso_reg(X_train, y_train, X_test, y_test):
    regr = Lasso(fit_intercept=False, alpha = .9, max_iter=100000)
    regr.fit(X_train, y_train)

    # project the estimate from elastic net fit to full D dimensions
    a_est = regr.coef_

    # compute normalized testing error
    train_error = norm_error(X_train, y_train, a_est)
    test_error = norm_error(X_test, y_test, a_est)
    return a_est, train_error, test_error

def PCA_elastic_net(X_train, y_train, X_test, y_test, K = 100):
    u, s, vh = np.linalg.svd(X_train, full_matrices = False)

    # reduce dimensions to the number of training samples
    X_prime = X_train @ vh.T[:,:K]

    # use elastic net regression from sklearn
    regr = ElasticNet(fit_intercept=False, l1_ratio = .5, alpha = .001, max_iter=1000000)
    regr.fit(X_prime, y_train)

    print(regr.coef_)
    # project the estimate from elastic net fit to full D dimensions
    a_est = vh.T[:,:K] @ regr.coef_

    # compute normalized testing error
    train_error = np.linalg.norm( X_train @ a_est - y_train) / np.linalg.norm(y_train)
    test_error = np.linalg.norm( X_test @ a_est - y_test ) / np.linalg.norm(y_test)
    return a_est, train_error, test_error

def exact_solution(X_train, y_train, X_test, y_test):
    a_est = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    train_error = np.linalg.norm( X_train @ a_est - y_train) / np.linalg.norm(y_train)
    test_error = np.linalg.norm( X_test @ a_est - y_test ) / np.linalg.norm(y_test)
    return a_est, train_error, test_error


## Regression
trials = 200
train_n = 100
test_n = 1000
d = 200

# part a)
results = np.zeros((2, trials))
for j in range(trials):
    X_train = np.random.normal(0, 1, size=(train_n, d))
    #a_true = np.random.normal(0, 1, size=(d, 1))
    a_true = np.random.normal(0,1, size=(d,1)) 
    y_train = X_train.dot(a_true) + np.random.normal(0, 0.5,size=(train_n, 1))
    X_test = np.random.normal(0, 1, size=(test_n,d))
    y_test = X_test.dot(a_true) + np.random.normal(0, 0.5,size=(test_n, 1))
    #a_est, train_error, test_error = PCA_compute_exact(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = Elastic_net(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = Lasso_reg(X_train, y_train, X_test, y_test)
    a_est, train_error, test_error = PCA_elastic_net(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = random_weights(X_train, y_train, X_test, y_test)
    results[0, j] =  train_error
    results[1, j] = test_error
print(results.mean(axis = 1))

# part b)
results = np.zeros((2, trials))
for j in range(trials):
    X_train = np.random.normal(0, 1, size=(train_n, d))
    a_true = np.random.normal(0,1, size=(d,1)) * np.random.binomial(1,0.1, size=(d,1))
    y_train = X_train.dot(a_true) + np.random.normal(0, 0.5,size=(train_n, 1))
    X_test = np.random.normal(0, 1, size=(test_n,d))
    y_test = X_test.dot(a_true) + np.random.normal(0, 0.5,size=(test_n, 1))
    #a_est, train_error, test_error = PCA_compute_exact(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = Elastic_net(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = Lasso_reg(X_train, y_train, X_test, y_test)
    a_est, train_error, test_error = PCA_elastic_net(X_train, y_train, X_test, y_test, K = 100)
    #a_est, train_error, test_error = random_weights(X_train, y_train, X_test, y_test)
    results[0, j] =  train_error
    results[1, j] = test_error
print(('train', 'test'))
print(results.mean(axis = 1))

