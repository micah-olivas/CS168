import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ego_fb = pd.read_csv("/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_6/cs168mp6.csv", header = None)
print(len(ego_fb))

friend_ids = set(ego_fb.iloc[:,0]) | set(ego_fb.iloc[:,1])
n = len(friend_ids)
print(n)

# Make Adjacency, Degree, and Laplacian matrices
A = np.zeros((n,n))
for _,row in ego_fb.iterrows():
    i,j = row
    A[i - 1, j - 1] = 1

D = np.diag(A.sum(axis = 0))
L = D - A

#Spectral decomposition of laplacian
d, v = np.linalg.eigh(L) # where d is array of eigenvalues, v is eigenvectors


#part 2b, first 12 eigenvalues
print([round(i,3) for i in d[:30]])

#part 2c, plot first 6 eigenvectors to find connnected components
fig, axs = plt.subplots(3, 2)
for i in range(6):
    axs[i % 3, i // 3].scatter(range(len(v)), v[:,i])
plt.tight_layout(pad=1)
plt.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_6/part2c_first6eigenvecs.png')

values, counts = np.unique(v[:,0].round(3), return_counts=True)
print(list(zip(values, counts)))

# part 2d, explore non-zero eigenvalue eigenvectors to find subsets with low conductance
plt.scatter(range(len(v)), v[:,6]) # plot singular values
plt.hlines(y = -.045, color='red', xmin=0,xmax=len(v), linestyles='dashed', label="S_1")
plt.hlines(y = 0.002, color = 'green', xmin=0, xmax=len(v), linestyles='dashed', label="S_2")
plt.hlines(y = -.044, color = 'green', xmin=0, xmax=len(v), linestyles='dashed')
plt.legend()
plt.savefig('part2d_eigenvector7.pdf')


def cond(A, S):
    """
        compute conductance of graph represented by adjacency matrix A
        on the subset of vertices specified in S
        input:
            A - 2d numpy array adjacency matrix
            S - 1d numpy array containing vertices of subset
        output:
            conductance of the subset S
    """
    V = np.array(range(len(A)))
    T = np.setdiff1d(V,S, assume_unique = True) #T is complement of S (V\S)
    return 2*A[S,:][:,T].sum() / min(A[S,:][:,S].sum(), A[T,:][:,T].sum())
print(v[:,6])
S_1 = np.array(np.where(v[:,6] < -.045))[0,:]
print(cond(A,S_1))

S_2 = np.array(np.where((v[:,6] < 0) & (v[:,6] >= -.4)))[0,:]
print(cond(A, S_2))

# find orthogonal set with low Conductance
def S3_search():

#part 1e, Null Conductance model
trials = 10000
null_cond = [ cond(A, np.random.choice(range(n), size=150, replace=False)) for _ in range(trials)]

plt.hist(null_cond, bins=100)
plt.xlabel('Conductance')
plt.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_6/proj6_part2e.null_conductance.png')

# null model had no subsets with cond < .1
print((np.array(null_cond) < 0.1).mean())
