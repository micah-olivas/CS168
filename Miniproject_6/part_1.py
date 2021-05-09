import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

# build adjaceny matrices and graph Laplacians
def build_mat1(n):
    D = 2*np.eye(n)
    D[0,0] = 1
    D[n-1,n-1] = 1
    A = np.eye(n, k=1) + np.eye(n, k=-1)
    L = D-A
    return L, A

def build_mat2(n):
    D = 3*np.eye(n)
    D[0,0] = 2
    D[n-2,n-2] = 2
    D[n-1,n-1] = n-1
    A = np.eye(n, k=1) + np.eye(n, k=-1)
    A[:,n-1] = 1
    A[n-1,:] = 1
    A[n-1,n-1] = 0
    L = D-A
    return L, A

def build_mat3(n):
    D = 2*np.eye(n)
    A = np.eye(n, k=1) + np.eye(n, k=-1)
    A[0,n-1] = 1
    A[n-1,0] = 1
    L = D-A
    return L, A

def build_mat4(n):
    D = 3*np.eye(n)
    D[n-1,n-1] = n-1
    A = np.eye(n, k=1) + np.eye(n, k=-1)
    A[0,n-2] = 1
    A[n-2,0] = 1
    A[:,n-1] = 1
    A[n-1,:] = 1
    A[n-1,n-1] = 0
    L = D-A
    return L, A

# define functions
def calc_and_sort_eigenvectors(mat):
    e_vals, e_vecs = LA.eig(mat)
    idx = e_vals.argsort()
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:,idx]
    return e_vecs

def plot_eigenvectors(e_vecs, ax, n, scatter=True):
    if scatter: 
        ax.scatter(range(len(e_vecs)), e_vecs[:,0], c='xkcd:red', label='k=1')
        ax.scatter(range(len(e_vecs)), e_vecs[:,1], c='xkcd:pink', label='k=2')
        ax.scatter(range(len(e_vecs)), e_vecs[:,n-2], c='xkcd:periwinkle', label='k=n-1')
        ax.scatter(range(len(e_vecs)), e_vecs[:,n-1], c='xkcd:indigo', label='k=n')
    else:
        ax.plot(range(len(e_vecs)), e_vecs[:,0], c='xkcd:red', label='k=1')
        ax.plot(range(len(e_vecs)), e_vecs[:,1], c='xkcd:pink', label='k=2')
        ax.plot(range(len(e_vecs)), e_vecs[:,n-2], c='xkcd:periwinkle', label='k=n-1')
        ax.plot(range(len(e_vecs)), e_vecs[:,n-1], c='xkcd:indigo', label='k=n')
    ax.legend()

def plot_spectral_embedding(e_vecs, ax, A, n, connect=True):
    for i in range(n):
        for j in range(i,n):
            if A[i,j] == 1:
                ax.scatter(L_vecs[[i,j],1], L_vecs[[i,j],2], c='k')
                if connect:
                    ax.plot(L_vecs[[i,j],1], L_vecs[[i,j],2], c='k')

def build_adjacency(xs, n, thresh):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if LA.norm(xs[i]-xs[j]) < thresh:
                A[i,j] = 1
                A[j,i] = 1
    D = np.diag(A.sum(axis=1))
    L = D-A
    return L,A  



# part b
fig, axs = plt.subplots(4,2, figsize=(10,12))

n = 100

for i in range(4):
    L, A = eval('build_mat{}(n)'.format(i+1))
    L_vecs = calc_and_sort_eigenvectors(L)
    plot_eigenvectors(L_vecs, axs[i,0], n, scatter=False)
    A_vecs = calc_and_sort_eigenvectors(A)
    plot_eigenvectors(A_vecs, axs[i,1], n, scatter=False)
    axs[i,0].set_ylabel('Graph {}'.format(i+1))
    
axs[0,0].set_title('L')
axs[0,1].set_title('A')
axs[3,0].set_xlabel('Coordinate')
axs[3,1].set_xlabel('Coordinate')

plt.tight_layout()
plt.savefig('figs/miniproject6.part1.b.png')
plt.clf()
            
# part c
fig, axs = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
ax_arr = axs.ravel()

n = 100

for i in range(4):
    L, A = eval('build_mat{}(n)'.format(i+1))
    L_vecs = calc_and_sort_eigenvectors(L)
    plot_spectral_embedding(L_vecs, ax_arr[i], A, n)
    ax_arr[i].set_title('Graph {}'.format(i+1))
axs[0,0].set_ylabel('Coordinate 3')
axs[1,0].set_ylabel('Coordinate 3')
axs[1,0].set_xlabel('Coordinate 2')
axs[1,1].set_xlabel('Coordinate 2')    

plt.tight_layout()
plt.savefig('figs/miniproject6.part1.c.png')
plt.clf()

# part d
n = 500
thresh = 0.25

xs = np.random.uniform(size=(n,2))
L,A = build_adjacency(xs, n, thresh)
L_vecs = calc_and_sort_eigenvectors(L)

idx = (xs[:,0] < 0.5) & (xs[:,1] < 0.5)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(L_vecs[idx,1],L_vecs[idx,2], c='red')
ax.scatter(L_vecs[~idx,1],L_vecs[~idx,2], c='k')

plt.tight_layout()
plt.savefig('figs/miniproject6.part1.d.png')
plt.clf()

