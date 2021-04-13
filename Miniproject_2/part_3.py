import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import importlib

from part_1 import *

### part 3
dat = pd.read_csv('Miniproject_2/data50.csv', sep = ',', names = ["articleId", "wordId", "Count"])
groups = pd.read_csv('Miniproject_2/p2_data/groups.csv', names = ['Name'])
labels = pd.read_csv('Miniproject_2/p2_data/label.csv', names = ['Label'])
#word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

# Sparse matrix implementation is slow, so store full matrix for now
word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()[1:,1:] # remove first row/col

## Locality-Sensitive-Hashing

# Hyperplane Hashing
def build_hashtables(v,d,k,l):
	"""
		Inputs:
		-v: a k-dimensional vector
		-d: lentgth of ith hashtable matrix
		-k: row dimension of ith hashtable matrix
		-l: number of hashtables
	"""
	hashtables = []

	for i in range(l):
		classification_matrix = np.zeros((d,k))

		with np.nditer(classification_matrix, op_flags=['readwrite']) as it:
		   for x in it:
			   x[...] = np.random.normal(0,1)

		hashtables.append(classification_matrix)

	# hash given vector v
	for tbl in range(l):
		hashtables[tbl] = hashtables[tbl].dot(v)
		with np.nditer(hashtables[tbl], op_flags=['readwrite']) as it:
			for x in it:
				if x > 0:
					x[...] = 1
				else:
					x[...] = 0

	return hashtables

v = np.array([2,6,3,5,8,2,4,3,4,6])

build_hashtables(v,10,10,10)

# Classification
def classify_hashed(q,tables):
	"""
		Inputs:
		-q: a vector with which to query the l hashtables for similarity
		-tables: a set of hastables for classification
	"""



v_hashtables = build_hashtables(v,d,k,l)
classify_hashed()

# implementation
l = 128
d = range(5,21)
