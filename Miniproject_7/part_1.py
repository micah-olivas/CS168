import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cycle_graph(n):
    '''
        create adjacency matrix and transition matrix for a cycle graph of n vertices
        input:
            - n : number of vertices in cycle graph
        returns:
            - A : numpy array representing adjacency matrix A[i, j] = 1 iff V_i ~ V_j
            - T : numpy array representing transition matrix of a random walk of graph 
    '''
    A = np.zeros((n,n))
    for i in range(n): A[i,(i-1)%n] = A[i,(i+1)%n] = 1
    D_inv = np.diag(A.sum(axis =0)**-1)
    T = D_inv @ A 
    return A, T

def get_s0(n):
    '''
        get initial state for a markov chain with n vertices
    '''
    return np.array([1] + [0]*(n-1))

## Create Cycle graph with 10 nodes
_, T_a = cycle_graph(10)

## Create cycle graph with 9 nodes
A, T_b = cycle_graph(9)

## Add edge to 9 node cycle graph
A[0,4] = A[4,0] = 1
D_inv = np.diag(A.sum(axis =0)**-1)
T_c = D_inv @ A


## Power Iteration algorithm to find stationary distribution
max_iterations = 1000
pi_a = get_s0(10)
pi_b = get_s0(9)
pi_c = get_s0(9)
for i in range(max_iterations):
    pi_a = pi_a @ T_a
    pi_b = pi_b @ T_b
    pi_c = pi_c @ T_c

print("Stationary Distributions:")
print("a. {}".format(pi_a))
print("b. {}".format(pi_b))
print("c. {}".format(pi_c))

# set stationary dit of 10 node cycle graph to uniform since it is aperiodic
pi_a = [1/10] * 10

# Calculate mixing times
s = get_s0(9)
mixing_time_b = 0
for i in range(max_iterations):
    s = s @ T_b
    if all(np.isclose(s, pi_b)): 
        mixing_time_b = i
        break
print("b. mixing time: {}".format(mixing_time_b))

s = np.array([1] + [0]*8)
mixing_time_c = 0
for i in range(max_iterations):
    s = s @ T_c
    if all(np.isclose(s, pi_c)): 
        mixing_time_c = i
        break
print("c. mixing time: {}".format(mixing_time_c))
    

# part b: total variation distance as a function of time parameter t
max_t = 100
s = get_s0(10)
tvd_a = [ .5*sum(abs((s := s @ T_a) - pi_a)) for i in range(max_t) ]
s = get_s0(9)
tvd_b = [ .5*sum(abs((s := s @ T_b) - pi_b)) for i in range(max_t) ]
s = get_s0(9)
tvd_c = [ .5*sum(abs((s := s @ T_c) - pi_c)) for i in range(max_t) ]
x = range(max_t)

plt.figure()
plt.plot(x, tvd_a, label = "a", color = 'red')
plt.plot(x, tvd_b, label = "b", color = 'blue')
plt.plot(x, tvd_c, label = "c", color = 'green')
plt.legend()
plt.savefig('mp7_part1b.png')

# extra graphs for part c
def calc_mixing_stats(T, n, t):
    #compute stationary dist with power iteration
    pi = get_s0(n)
    for i in range(1000):
        pi = pi @ T

    s = get_s0(n)
    mixing_time = 0
    for i in range(1000):
        s = s @ T
        if all(np.isclose(s, pi)): 
            mixing_time = i
            break

    d, u = np.linalg.eig(T)
    d.sort()
    second_eigval = d[-2]

    s = get_s0(n)
    tvd = [ .5*sum(abs((s := s @ T) - pi)) for i in range(t) ]
    return mixing_time, second_eigval, tvd

n = 21
TVD = np.zeros((n-1, max_t))
mixing_times = np.zeros((n-1))
second_eigvals = np.zeros((n-1))
A, T = cycle_graph(n)

for i in range(n-1):
    A[0, 1+i] = A[1+i, 0] = 1
    D_inv = np.diag(A.sum(axis =0)**-1)
    T = D_inv @ A
    mixing_time, second_eigval, tvd = calc_mixing_stats(T, n, max_t)

    TVD[i,:] = tvd
    mixing_times[i] = mixing_time
    second_eigvals[i] = second_eigval

mixing_df = pd.DataFrame({'mixing_time':mixing_times, 'second_eigval':second_eigvals,
    'extra_edges':range(n-1)})
plt.figure()
sns.scatterplot(data=mixing_df, x='mixing_time', y='second_eigval', hue='extra_edges', palette='OrRd')
plt.savefig('mp7_part1c.eigval_mixing_time.scatter.png')

TVD_df = pd.DataFrame({ '+' + str(i) : TVD[i] for i in range(n-1) if i % 2 == 0})
plt.figure()
sns.lineplot(data=TVD_df, palette="OrRd")
plt.savefig('mp7_part1c.20_cycle_tvd.png')
