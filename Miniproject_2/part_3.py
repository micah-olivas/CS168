import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import importlib

# from part_1 import *

# Previous functions
def cosine_similarity(a, b):
    """
        inputs:
            -a : a csr sparse vector
            -b : another csr sparse vector
        returns:
            the cosine similarity of the two sparse vectors
    """
    #return (a @ b.T)[0,0] / (sparse_norm(a)*sparse_norm(b))
    return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))

### part 3
dat = pd.read_csv('Miniproject_2/data50.csv', sep = ',', names = ["articleId", "wordId", "Count"])
groups = pd.read_csv('Miniproject_2/p2_data/groups.csv', names = ['Name'])
labels = pd.read_csv('Miniproject_2/p2_data/label.csv', names = ['Label'])
#word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

# Sparse matrix implementation is slow, so store full matrix for now
word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()[1:,1:] # remove first row/col

## Locality-Sensitive-Hashing

# Construct hyperplane hashtables
def build_hashtables(d,k,l):
	"""
		Inputs:
		-d: lentgth of ith hashtable matrix
		-k: row dimension of ith hashtable matrix
		-l: number of hashtables
	"""
	matrix_array = []

	for i in range(l):
		classification_matrix = np.zeros((d,k))

		with np.nditer(classification_matrix, op_flags=['readwrite']) as it:
		   for x in it:
			   x[...] = np.random.normal(0,1)

		matrix_array.append(classification_matrix)

	return matrix_array

a = build_hashtables(5,5,5)
a

def hash_vector(v,matrix_array):
	"""
		Inputs:
		-v: k-dimensional vector to hash
		-matrix_array: array of matrices containing values randomly sampled from gaussian with u = 0, var = 1
	"""
	hashtables = matrix_array

	# hash given vector v
	for tbl in range(len(hashtables)):
		hashtables[tbl] = hashtables[tbl].dot(v)
		with np.nditer(hashtables[tbl], op_flags=['readwrite']) as it:
			for x in it:
				if x > 0:
					x[...] = 1
				else:
					x[...] = 0

	return hashtables

b = np.array([0,6,3,4,5])
c = hash_vector(b,a)
c
a
# Hash an array of vectors
def hash_vectors(v_array,d,k,l):
	"""
		Inputs:
		-v_array: an array dataset of k-dimensional vectors
	"""
	hashed_array = []

	for vec in v_array:
		v_array[vec] = build_hashtables(vec,d,k,l)

	return dict(zip(list(v_array), hashed_array))

# Classification
def brute_force_search(query, search_space, metric=cosine_similarity, self=0):
    similarity_vec = np.apply_along_axis(lambda row: metric(row, query), 1, search_space)
    similarity_vec[0,self] = np.NINF
    max_idx = np.nanargmax(similarity_vec)
    return max_idx

def classify_query(q,d,k,l,tables):
	"""
		Inputs:
		-q: a query vector
		-tables: a set of hastables for classification
	"""

	# hash query data, assuming hashtables for
	similar_obj = brute_force_search(q, tables)
	sim_obj_label = labels.loc[sim_obj, 'Label']

def build_classification_matrix(data_mat, groups=groups, labels=labels):
    classification_matrix = np.zeros((len(groups),len(groups)))

    for i in range(len(labels)):
        true_label = labels.loc[i,'Label']
        doc = data_mat[i]
        most_sim_doc = brute_force_search(doc, data_mat, self=i)
        most_sim_label = labels.loc[most_sim_doc,'Label']
        classification_matrix[true_label-1,most_sim_label-1] += 1

    return classification_matrix


v_hashtables = build_hashtables(v,d,k,l)
classify_hashed()

#

## 3.c
l = 128

# compute classification error with d
def compute_classification_accuracy(classification_mat):
    return classification_mat.diagonal().sum() / classification_mat.sum()

class_accuracies = []

for d_i in range(5,21):
	hashtables_di = build_hashtables(d_i,)
	class_di = classify_hashed()
	class_accuracies.append(compute_classification_accuracy())

plt.plot(range(5,21), class_accuracies, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

a = range(5,21)
b = range(5,21)

plt.plot(a, b, 'bo', linestyle='dashed', linewidth=2, markersize=8)
