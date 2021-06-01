import cvxpy as cvx
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt

## part a
x = []
with open('wonderland-tree.txt') as infile:
    for line in infile:
        x.extend(map(int, list(line.strip())))
x = np.array(x)
n = x.size
print("n = {}".format(n))
k = sum(x == 1)
print("k = {}".format(k))
print("n/k = {}".format(n/k))

## part b
r = 600
A = np.random.normal(0,1, (n, n))
A_r = A[:r,:]

b = A_r @ x

vx = cvx.Variable(n)
vy = cvx.Variable(n)
objective = cvx.Minimize(cvx.sum(vy))
constraints = [A_r @ vx == b, vy - vx >= 0, vy + vx >= 0, vy >= 0, vx >= 0]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

assert np.allclose(x, vx.value)


## part c
def compressive_sensing_recover(A, r, x):
    A_r = A[:r,:]
    b = A_r @ x
    vx = cvx.Variable(n)
    vy = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum(vy))
    constraints = [A_r @ vx == b, vy - vx >= 0, vy + vx >= 0, vy >= 0, vx >= 0]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    return vx.value

lo = 0
hi = 600
while lo < hi:
    r = (hi + lo) // 2
    print(r)
    x_r = compressive_sensing_recover(A, r, x)
    if np.linalg.norm(x - x_r,1) < .001:
        hi = r - 1
    else:
        lo = r + 1

r_star = lo
print(r_star)

## part d
reconstruction_error = [ np.linalg.norm(compressive_sensing_recover(A, r_star + i, x) - x, 1) for i in range(-10,3)]

x_plot = r_star + np.array(range(-10,3))
plt.plot(x_plot, reconstruction_error)
plt.savefig('mp9.part1.d.png')