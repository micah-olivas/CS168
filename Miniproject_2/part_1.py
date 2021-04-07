import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import itertools

# Heatmap
def makeHeatMap(data, names, color, similarity_metric, outputFileName):
            #code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
            fig, ax = plt.subplots()
            #create the map w/ color bar legend
            heatmap = ax.pcolor(data, cmap=color)
            cbar = plt.colorbar(heatmap)

            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
            ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            ax.set_xticklabels(names)
            plt.xticks(rotation=90)
            ax.set_yticklabels(names)

            ax.set_title(similarity_metric)
            plt.tight_layout()

            plt.savefig(outputFileName, format = 'png')
            plt.close()

### part 1

def jaccard_similarity(mat):
    """
        inputs: 
            -mat : a 2 row matrix 
        returns:
            Jaccard similarity of the 2 rows
    """
    return mat.min(axis=0).sum() / mat.max(axis=0).sum()

def sparse_norm(v):
    """
        inputs:
            -v : a csr sparse representation of a 1-d vector
        returns:
            the norm of the vector
            (should be indentical to np.linalg.norm)
    """
    return np.sqrt(v @ v.T)[0,0]

def L2_similarity(a, b):
    """
        inputs:
            -a : a csr sparse vector
            -b : another csr sparse vector 
        returns:
            the L2 similarity of the two sparse vectors
    """
    return -sparse_norm(a - b)

def cosine_similarity(a, b):
    """
        inputs:
            -a : a csr sparse vector
            -b : another csr sparse vector 
        returns:
            the cosine similarity of the two sparse vectors
    """
    return (a @ b.T)[0,0] / (sparse_norm(a)*sparse_norm(b))

dat = pd.read_csv('data50.csv', sep = ',', names = ["articleId", "wordId", "Count"])
groups = pd.read_csv('p2_data/groups.csv', names = ['Name'])
labels = pd.read_csv('p2_data/label.csv', names = ['Label'])
#word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

# Sparse matrix implementation is slow, so store full matrix for now
word_mat = sparse.csr_matrix((dat.Count, (dat.articleId, dat.wordId))).todense()

similarity_matrix = np.zeros((3,len(groups),len(groups)))
for i,j in itertools.combinations_with_replacement(groups.index, 2):
    # row number in file is 0-indexed but labels and articles are 1-indexed so we add 1 
    articles_i = labels[labels.Label == i + 1].index + 1
    articles_j = labels[labels.Label == j + 1].index + 1
    pairwise_results = np.zeros((3, len(articles_i)*len(articles_j)))
    k = 0
    for a,b in itertools.product(articles_i, articles_j):
        # TODO: debug why implementation with sparse matrix is so inefficient
#        pairwise_results[0, k] = jaccard_similarity(word_mat[[a, b]])
#        pairwise_results[1, k] = L2_similarity(word_mat[a], word_mat[b])
#        pairwise_results[2, k] = cosine_similarity(word_mat[a], word_mat[b])
        # jaccard
        pairwise_results[0, k] = word_mat[[a, b]].min(axis=0).sum() / word_mat[[a,b]].max(axis=0).sum()
        # L2
        pairwise_results[1, k] = -np.linalg.norm(word_mat[a] - word_mat[b])
        # cosine
        pairwise_results[2, k] = (word_mat[a] @ word_mat[b].T) / (np.linalg.norm(word_mat[a])*np.linalg.norm(word_mat[b]))
        k += 1
    similarity_matrix[:,i,j] = similarity_matrix[:,j,i] = pairwise_results.mean(axis = 1)

makeHeatMap(similarity_matrix[0], groups.Name, 'magma', 'Jaccard Similarity', 'figs/part1_jaccard.png')
makeHeatMap(similarity_matrix[1], groups.Name, 'magma', 'L2 Similarity', 'figs/part1_l2.png')
makeHeatMap(similarity_matrix[2], groups.Name, 'magma', 'Cosine Similarity', 'figs/part1_cosine.png')

