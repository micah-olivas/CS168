import numpy as np

# Warm up

X = np.matrix(np.arange(0.001, 1, 0.001)) # BUG: some steps are slightly longer than 0.001; shouldnt be a concern though
Y = X*2

def pca_recover():
    print('o')
