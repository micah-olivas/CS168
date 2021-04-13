import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import itertools

from part_1 import *

### part 3

dat = pd.read_csv('Miniproject_2/data50.csv', sep = ',', names = ["articleId", "wordId", "Count"])
groups = pd.read_csv('Miniproject_2/p2_data/groups.csv', names = ['Name'])
labels = pd.read_csv('Miniproject_2/p2_data/label.csv', names = ['Label'])
#word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

# Sparse matrix implementation is slow, so store full matrix for now
word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()[1:,1:] # remove first row/col

# Locality-Sensitive-Hashing

# Hyperplane Hashing
def build_hashtables(d,k,l):
	"""
		Inputs:
		-d:
		-k: dimensionality of row vector in matrix
		-l: number of hashtables
	"""
	hashtables = []

	for i in range(l):
		classification_matrix = np.zeros((d,k))

		with np.nditer(classification_matrix, op_flags=['readwrite']) as it:
		   for x in it:
			   x[...] = np.random.normal(0,1)
			   x[...] = np.sign(x)

		hashtables.append(classification_matrix)

	return hashtables

def classify_hashed(a):
	"""
		Inputs:
		-q: query vector

	"""


# Classification
