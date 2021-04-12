import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import itertools

from part_1 import *

### part 2

dat = pd.read_csv('data50.csv', sep = ',', names = ["articleId", "wordId", "Count"])
groups = pd.read_csv('p2_data/groups.csv', names = ['Name'])
labels = pd.read_csv('p2_data/label.csv', names = ['Label'])
#word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

# Sparse matrix implementation is slow, so store full matrix for now
word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()[1:,1:] # remove first row/col

def brute_force_search(query, search_space, metric=cosine_similarity, self=0):
    similarity_vec = np.apply_along_axis(lambda row: metric(row, query), 1, search_space)
    similarity_vec[0,self] = np.NINF
    max_idx = np.nanargmax(similarity_vec)
    return max_idx

def build_classification_matrix(data_mat, groups=groups, labels=labels):
    classification_matrix = np.zeros((len(groups),len(groups)))

    for i in range(len(labels)):
        true_label = labels.loc[i,'Label']
        doc = data_mat[i]
        most_sim_doc = brute_force_search(doc, data_mat, self=i)
        most_sim_label = labels.loc[most_sim_doc,'Label']
        classification_matrix[true_label-1,most_sim_label-1] += 1

    return classification_matrix

def compute_classification_accuracy(classification_mat):
    return classification_mat.diagonal().sum() / classification_mat.sum()

baseline_classification_matrix = build_classification_matrix(word_mat)
makeHeatMap(baseline_classification_matrix, groups.Name, 'magma', 'Baseline Classification', 'figs/part2.baseline_classification.png')
print(compute_classification_accuracy(baseline_classification_matrix))

ds = [10, 25, 50, 100, 200, 500, 1000]
k = word_mat.shape[1]

dim_red_classification_matrix = np.zeros((len(ds),len(groups),len(groups)))

for idx, d in enumerate(ds):
    random_mat = np.random.normal(size=(k,d))
    projected_mat = word_mat @ random_mat
    classification_mat = build_classification_matrix(projected_mat)
    dim_red_classification_matrix[idx] = classification_mat

for idx, d in enumerate(ds):
    print(d)
    print(compute_classification_accuracy(dim_red_classification_matrix[idx]))
    makeHeatMap(dim_red_classification_matrix[idx], groups.Name, 'magma', 'Dimensionality Reduction Classification (d={})'.format(d), 'figs/part2.dim_red_classification.{}.png'.format(d))
