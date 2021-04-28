import numpy as np

# Warm up

X = np.array(np.arange(0.001, 1, 0.001)) # BUG: some steps are slightly longer than 0.001; shouldnt be a concern though
Y = X*2

def pca_recover():

    print('o')

def ls_recover(X, Y):
    np.linalg.inner(X, Y)
