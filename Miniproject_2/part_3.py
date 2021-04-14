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
def build_empty_hashtables(d,k,l):
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

def hash_vector(v,matrix_array):
	"""
		Inputs:
		-v: k-dimensional vector to hash
		-matrix_array: from build_empty_hashtables; array of matrices containing values randomly sampled from gaussian with u = 0, var = 1
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

a = np.array([0,1,1,1,0])

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

### ben stuff

l = 128
ds = list(range(5,21))
k = word_mat.shape[1]

def compute_hash_val(v, mat):
    sign_vec = np.sign(v @ mat)
    return sign_vec_to_int(sign_vec)

def sign_vec_to_int(sign_vec):
    return int(''.join(map(str,(.5*(sign_vec+1)).astype(np.int).tolist()[0])),2)

def build_hashtables(dat_mat, d, k=k, l=l, n=1000):
    random_mats = [np.random.normal(size=(k,d)) for _ in range(l)]
    hash_tables = [{} for _ in range(l)]
    for j in range(n):
        v = dat_mat[j]
        for i in range(l):
            mat = random_mats[i]
            hash_i = compute_hash_val(v, mat)
            curr_val = hash_tables[i].get(hash_i, [])
            hash_tables[i][hash_i] = curr_val + [j]
    return random_mats, hash_tables

def query(q, mats, hash_tables):
    possible_matches = []
    for i in range(l):
        mat = mats[i]
        hash_i = compute_hash_val(q, mat)
        possible_matches += hash_tables[i].get(hash_i, [])
    return set(possible_matches)

def build_classification_matrix(data_mat, mats, hash_tables, groups=groups, labels=labels):
    classification_matrix = np.zeros((len(groups),len(groups)))

    Sqs = []

    for i in range(len(labels)):
        true_label = labels.loc[i,'Label']
        doc = data_mat[i]
        possible_docs = query(doc, mats, hash_tables)
        possible_docs = list(possible_docs)
        Sqs.append(len(possible_docs)-1)
        subset_data_mat = data_mat[possible_docs]
        subset_labels = labels.loc[possible_docs]
        self_idx = possible_docs.index(i)
        most_sim_doc = brute_force_search(doc, subset_data_mat, self=self_idx)
        most_sim_label = subset_labels.iloc[most_sim_doc,0]
        classification_matrix[true_label-1,most_sim_label-1] += 1

    return classification_matrix, np.mean(Sqs)


class_mat = build_classification_matrix(word_mat, random_mats, hash_tables)

## 3.b
l = 177.5
d = 134.4658
k = 1
N = 1000000

l*k*d
k*N*(1 - (1 - (1 - .2/3.14159)**d)**l)

## 3.c
# compute classification error at range of d vals
def compute_classification_accuracy(classification_mat):
    return classification_mat.diagonal().sum() / classification_mat.sum()

def plot_classification_errors(dat_mat):
	class_accuracies = []

	for d_i in range(5,21):
		hashtables_empty_di, hashtables_d_i = build_hashtables(dat_mat, d_i, k, l)
		class_di = build_classification_matrix()
		class_accuracies.append(compute_classification_accuracy(hashtables_d_i, class_di))

	errors_plot = plt.plot(range(5,21), class_accuracies, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
	return errors_plot

##
