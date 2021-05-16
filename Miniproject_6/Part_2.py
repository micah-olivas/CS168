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
plt.scatter(range(len(v)), v[:,6]) # plot
plt.fill_between([0,len(v)], [-.04], [-0.12], color='red', alpha=0.1, label='S_1')
plt.fill_between([0,len(v)], [-.04], [0], color='green', alpha=0.1, label='S_2')
plt.fill_between([0,len(v)], [0.01335], [0.002], color='purple', alpha=0.2, label='S_3')
plt.legend(bbox_to_anchor=(-.1,.5))
plt.tight_layout()
plt.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_6/proj6_part2d_eigenvector7.png')


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
    # return 2*A[S,:][:,T].sum() / min(A[S,:][:,S].sum(), A[T,:][:,T].sum())
    return A[S,:][:,T].sum() / min(A[S,:].sum(), A[T,:].sum()) # ammended conductance formula

S_1 = np.array(np.where(v[:,6] < -.04))[0,:]
index = np.random.choice(S_1.shape[0], 10, replace=False)
print(round(cond(A,S_1), 4), len(S_1), S_1[index])

S_2 = np.array(np.where((v[:,6] < 0) & (v[:,6] >= -.4)))[0,:]
index = np.random.choice(S_2.shape[0], 10, replace=False)
print(round(cond(A, S_2), 4), len(S_2), S_2[index])

S_3 = np.array(np.where((v[:,6] < 0.01335) & (v[:,6] >= 0.01)))[0,:]
index = np.random.choice(S_3.shape[0], 10, replace=False)
print(round(cond(A,S_3), 4), len(S_3), S_3[index])

#part 1e, Null Conductance model
trials = 10000
S_rand = np.random.choice(range(n), size=150, replace=False)

cond(A, S_rand)

null_cond = [cond(A, np.random.choice(range(n), size=150, replace=False)) for _ in range(trials)]
np.mean(null_cond)
plt.hist(null_cond, bins=100)
plt.xlabel('Conductance')
plt.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_6/proj6_part2e.null_conductance.png')

# null model had no subsets with cond < .1
print((np.array(null_cond) < 0.1).mean())
