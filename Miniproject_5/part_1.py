import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 1.a Confirming processing
co_occur = pd.read_csv('Miniproject_5/co_occur.csv', sep = ',', names = list(range(10000)))

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
plt.savefig('Miniproject_5/mp5-part1b.png', dpi = 300)

# 1.c
u
