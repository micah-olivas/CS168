import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 1.a Confirming co-occur, dictionary processing
co_occur = pd.read_csv('Miniproject_5/co_occur.csv', sep = ',', names = list(range(10000)))
with open('Miniproject_5/dictionary.txt') as f:
    content = f.readlines()
dict = [x.strip() for x in content]

# 1.b Normalize matrix and compute rank-100 approx
def norm_matrix(elem):
    elem = math.log(elem + 1)
    return elem

norm_co_occur = co_occur.applymap(norm_matrix)
top_co_occur = norm_co_occur.iloc[:3000, :3000]

# Single Value Decomposition
u, s, vh = np.linalg.svd(top_co_occur)
s

# Plot singular values
plt.scatter(y = s, x = list(range(len(s))))
plt.ylabel('Singular Value')
plt.show()
plt.savefig('Miniproject_5/mp5-part1b.png', dpi = 300)

# 1.c index single

# Get co-occurrence values from word pair
def words_co_occurence(word1, word2):
    """
        inputs:
            -word1 : first word
            -word2 : second word
        returns:
            score from the co-occurrence matrix corresponding
            to the intersection of word1 and word2
    """
    word1 = str(word1)
    word2 = str(word2)

    idx1 = dict.index(word1)
    idx2 = dict.index(word2)

    co_occurrence = norm_co_occur.iloc[idx1,idx2]

    return co_occurrence

words_co_occurence("chicago", "sacramento")

# 1.d.i
v_words = ["boy", "girl", "brother", "sister", "king", "queen", he, she, john, mary, wall, tree]

# 1.e.i
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


vec1 = [1,2,3,4,5]
vec2 = [2,3,4,5,6]
vec3 = [9,9,9,9,9]
