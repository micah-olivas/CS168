import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

### Part 2

train_n = 100
test_n = 1000
d = 100

def generate_data():
    X_train = np.random.normal(0,1, size=(train_n,d))
    a_true = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
    
    return X_train, a_true, y_train, X_test, y_test

def solve_exact(X,y):
    a_hat = np.linalg.inv(X).dot(y)
    return a_hat

def normalized_error(X,a,y):
    return np.linalg.norm(X.dot(a)-y) / np.linalg.norm(y)

# part a
n_trials = 10
train_errs, test_errs = [], []

for _ in range(n_trials):
    X_train, a_true, y_train, X_test, y_test = generate_data()
    a_hat = solve_exact(X_train, y_train)
    train_err, test_err = normalized_error(X_train,a_hat,y_train), normalized_error(X_test,a_hat,y_test)
    train_errs.append(train_err)
    test_errs.append(test_err)

print('part a')
print(np.mean(train_errs), np.mean(test_errs))

# part b
def solve_exact_l2(X,y,l,d=d):
    a_hat = np.linalg.inv(np.transpose(X).dot(X) + l*np.identity(d)).dot(np.transpose(X).dot(y))
    return a_hat

n_trials = 10
ls = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
data = []

for l in ls:
    for _ in range(n_trials):
        X_train, a_true, y_train, X_test, y_test = generate_data()
        a_hat = solve_exact_l2(X_train, y_train, l)
        train_err, test_err = normalized_error(X_train,a_hat,y_train), normalized_error(X_test,a_hat,y_test)
        data.append({'l2':l,'Error':train_err,'Error_type':'train'})
        data.append({'l2':l,'Error':test_err,'Error_type':'test'})


df = pd.DataFrame(data)
sns.barplot(data=df,x='l2',y='Error',hue='Error_type')
plt.savefig('figs/miniproject3.part2.b.png')
plt.clf()

# part c
def error_gradient(X,a,y):
    return (2*(X.dot(a)-y)*X).reshape(-1,1)

def sgd(a_guess, X, y, alpha, n=train_n, n_iter=1000000):
    a_hat = a_guess
    for i in range(n_iter):
        idx = np.random.randint(n)
        x_i = X[idx]
        y_i = y[idx]
        grad = error_gradient(x_i, a_hat, y_i)
        # before = error(x_i,a_hat,y_i)
        a_hat = a_hat - alpha * grad
        # after = error(x_i,a_hat,y_i)
        # assert before > after
    return a_hat

n_trials = 10
alphas = [0.00005, 0.0005, 0.005]
data = []

for alpha in alphas:
    for _ in range(n_trials):
        X_train, a_true, y_train, X_test, y_test = generate_data()
        a_guess = np.zeros((d,1))
        a_hat = sgd(a_guess, X_train, y_train, alpha)
        train_err, test_err = normalized_error(X_train,a_hat,y_train), normalized_error(X_test,a_hat,y_test)
        train_err_true, test_err_true = normalized_error(X_train,a_true,y_train), normalized_error(X_test,a_true,y_test)
        data.append({'alpha':alpha,'Error':train_err,'Error_type':'train'})
        data.append({'alpha':alpha,'Error':test_err,'Error_type':'test'})
        data.append({'alpha':alpha,'Error':train_err_true,'Error_type':'train_true'})
        data.append({'alpha':alpha,'Error':test_err_true,'Error_type':'test_true'})

df = pd.DataFrame(data)
sns.barplot(data=df,x='alpha',y='Error',hue='Error_type')
plt.savefig('figs/miniproject3.part2.c.png')
plt.clf()

# part d
def augmented_sgd(a_guess, X, y, alpha, n=train_n, n_iter=1000000, sampling_rate=100):
    a_hat = a_guess
    a_arr = np.zeros((a_guess.size, int(n_iter/sampling_rate)))
    for i in range(n_iter):
        idx = np.random.randint(n)
        x_i = X[idx]
        y_i = y[idx]
        grad = error_gradient(x_i, a_hat, y_i)
        a_hat = a_hat - alpha * grad
        if i % sampling_rate == 0:
            a_arr[:,int(i/sampling_rate)] = a_hat.reshape(-1)
    return a_hat, a_arr

alphas = [0.00005, 0.005]
data = []
sr = 100
n_iter = 1000000

fig, axs = plt.subplots(3,2, figsize=(7,7))

for idx, alpha in enumerate(alphas):
    X_train, a_true, y_train, X_test, y_test = generate_data()
    a_guess = np.zeros((d,1))
    a_hat, a_arr = augmented_sgd(a_guess, X_train, y_train, alpha, n_iter=n_iter, sampling_rate=sr)
    axs[0,idx].plot(range(n_iter//sr),[normalized_error(X_train,a_arr[:,i].reshape(-1,1),y_train) for i in range(n_iter//sr)])
    axs[0,idx].axhline(normalized_error(X_train,a_true,y_train), ls='--')
    if idx == 0:
        axs[0,idx].set_ylabel('Normalized training error')
    axs[0,idx].set_title(alpha)
    axs[1,idx].plot(range(n_iter//sr),[normalized_error(X_test,a_arr[:,i].reshape(-1,1),y_test) for i in range(n_iter//sr)])
    axs[1,idx].axhline(normalized_error(X_test,a_true,y_test), ls='--')
    if idx == 0:
        axs[1,idx].set_ylabel('Normalized test error')
    axs[2,idx].plot(range(n_iter//sr),[np.linalg.norm(a_arr[:,i].reshape(-1,1)) for i in range(n_iter//sr)])
    axs[2,idx].set_xlabel('Iteration (x {})'.format(sr))
    if idx == 0:
        axs[2,idx].set_ylabel('Norm(a)')

plt.savefig('figs/miniproject3.part2.d.png')
plt.clf()

# part e
def sample_from_unit_ball(r, d=d):
    v = np.random.normal(0,1,size=(d,1))
    return v / np.linalg.norm(v) * r

alpha = 0.00005
rs = [0, 0.1, 0.5, 1, 10, 20, 30]
n_trials = 10

data = []

for r in rs:
    for _ in range(n_trials):
        X_train, a_true, y_train, X_test, y_test = generate_data()
        a_guess = sample_from_unit_ball(r)
        a_hat = sgd(a_guess, X_train, y_train, alpha)
        train_err, test_err = normalized_error(X_train,a_hat,y_train), normalized_error(X_test,a_hat,y_test)
        data.append({'r':r,'Error':train_err,'Error_type':'train'})
        data.append({'r':r,'Error':test_err,'Error_type':'test'})

df = pd.DataFrame(data)
sns.barplot(data=df,x='r',y='Error',hue='Error_type')
plt.savefig('figs/miniproject3.part2.e.png')
plt.clf()


