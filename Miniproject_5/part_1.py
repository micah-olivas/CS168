import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 1.a Confirming processing
co_occur = pd.read_csv('Miniproject_5/co_occur.csv', sep = ',', names = list(range(10000)))

## 1.b Normalize matrix and compute rank-100 approx
def norm_matrix(elem):
    elem = math.log(elem + 1)
    return elem

norm_co_occur = co_occur.applymap(norm_matrix)
top_co_occur = norm_co_occur.iloc[:100, :100]

# Single Value Decomposition
u, s, vh = np.linalg.svd(norm_co_occur)
s

W = top_co_occur @ np.transpose(top_co_occur)
W

# Plot single values
plt.stem(s)
