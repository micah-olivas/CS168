import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft

# a
p = [1/6,1/6,1/6,1/6,1/6,1/6]
def conv(x, y):
    if isinstance(x, list) == False:
        x = np.array(list(x))
    if isinstance(y, list) == False:
        y = np.array(list(y))

    return np.fft.ifft(np.multiply(fft(x),fft(y)))

conv(p, p)

# b
def f_plus(f):
    N = len(f)
    f_plus = np.zeros(2*N)
    for i in range(2*N):
        if (i >= 0) and (i <= N-1):
            f_plus[i] = f[i]
        if (i >= N) and (i <= 2*N-1):
            f_plus[i] = 0

    return f_plus

fft()

# c
def multiply(x, y):
    if isinstance(x, list) == False:
        x = np.array(list(x))
    if isinstance(y, list) == False:
        y = np.array(list(y))

    conv = np.fft.ifft(np.multiply(fft(x),fft(y)))

# d
def grade_school_interp(x, y):


if 0 < 0:
    print("H")

x = [1,2,3,4]
np.array(x)
