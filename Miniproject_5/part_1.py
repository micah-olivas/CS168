import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import os.path
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

# Single Value Decomposition
if os.path.exists('Miniproject_5/svd_d.csv'): # if svd ouput exists in dir, don't recalculate
    u = pd.read_csv('Miniproject_5/svd_u.csv', sep = ',', index_col=0)
    d = pd.read_csv('Miniproject_5/svd_d.csv', sep = ',', index_col=0)
    v = pd.read_csv('Miniproject_5/svd_v.csv', sep = ',', index_col=0)

else:
    u, d, v = np.linalg.svd(norm_co_occur)
    u = pd.DataFrame(u)
    d = pd.DataFrame(d)
    v = pd.DataFrame(v)
    u.to_csv('Miniproject_5/svd_u.csv')
    d.to_csv('Miniproject_5/svd_d.csv')
    v.to_csv('Miniproject_5/svd_v.csv')

# Plot singular values
plt.scatter(y = d, x = list(range(len(d))))
plt.ylabel('Singular Value')
plt.show()
plt.savefig('Miniproject_5/mp5-part1b.png', dpi = 300)

# 1.c index single word

def words_to_idxs(word_list):
    return [dict.index(i) for i in word_list]
def idxs_to_words(idx_list):
    return [dict[i] for i in idx_list]


def get_sing_vec(i):
    """
        inputs:
            -i: the index of a word in matrix M
        returns:
            the largest value in a singular vector
    """
    word = idxs_to_words([i])[0]

    i = str(i)
    head_val = 10

    # get and sort vi
    vi = v[[i]]
    vhi = vi.sort_values(ascending=False, by=[i]).head(head_val).round(decimals=4)
    vhi_idxs = list(vhi.index)

    vlow = vi.sort_values(ascending=True, by=[i]).head(head_val).round(decimals=4)
    vlow_idxs = list(vlow.index)

    words_hi = idxs_to_words(vhi_idxs)
    words_low = idxs_to_words(vlow_idxs)


    vhi['Highly Associated Word'] = words_hi
    vhi = vhi.rename(columns={i:"High Co-occurrence"})

    vlow['Lowly Associated Word'] = words_low
    vlow = vlow.rename(columns={i:"Low Co-occurrence"})

    v_all = pd.concat([vhi,vlow], ignore_index=False, axis=1)
    # vi.rename(mapper=)
    # label_changes = dict(zip([range(head_val)], words))

    return vhi, vlow

# define list of words for tables
word_list = ["lay", "strong", "piece", "work", "signal"]

def list_to_tables(my_list):
    for i in my_list:
        idx = words_to_idxs([i])
        df_hi, df_low  = get_sing_vec(idx[0])
        df_hi.to_csv('Miniproject_5/part_1c/'+ str(i) + '_hi.csv')
        df_low.to_csv('Miniproject_5/part_1c/'+ str(i) + '_low.csv')


list_to_tables(word_list)


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

# 1.d.i
v_words = ["boy", "girl", "brother", "sister", "king", "queen", "he", "she", "john", "mary", "wall", "tree"]

np.linalg.norm(u[['1']])
u[[1]]



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


cosine_similarity(vec2,vec4)
