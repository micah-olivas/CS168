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
#naive grade school implementation
def multiply(A, B, m, n):

    prod = [0] * (m + n - 1);

    # Multiply two polynomials term by term

    # Take ever term of first polynomial
    for i in range(m):

        # Multiply the current term of first
        # polynomial with every term of
        # second polynomial.
        for j in range(n):
            prod[i + j] += A[i] * B[j];

    return prod;

# A utility function to print a polynomial
def printPoly(poly, n):

    for i in range(n):
        print(poly[i], end = "");
        if (i != 0):
            print("x^", i, end = "");
        if (i != n - 1):
            print(" + ", end = "");

A = [5, 0, 10, 6];

# The following array represents
# polynomial 1 + 2x + 4x^2
B = [1, 2, 4];
m = len(A);
n = len(B);

print("First polynomial is ");
printPoly(A, m);
print("\nSecond polynomial is ");
printPoly(B, n);

prod = multiply(A, B, m, n);

print("\nProduct polynomial is ");
printPoly(prod, m+n-1);
