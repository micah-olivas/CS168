import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pylab import cm

### Part 2

# load matrix
one_thousand_genomes = pd.read_table('p4dataset2021.txt', header=None, sep='\\s+', index_col=0)

sexes = one_thousand_genomes[1]
populations = one_thousand_genomes[2]

sex_dict = {1: 'MALE', 2: 'FEMALE'}

pop_dict = {'ACB': 'African Caribbeans in Barbados', 
            'GWD': 'Gambian in Western Divisions in the Gambia', 
            'ESN': 'Esan in Nigeria', 
            'MSL': 'Mende in Sierra Leone', 
            'YRI': 'Yoruba in Ibadan, Nigeria', 
            'LWK': 'Luhya in Webuye, Kenya', 
            'ASW': 'Americans of African Ancestry in SW USA'}

geno_columns = range(3,10104)
genotypes = one_thousand_genomes[range(3,10104)]

# get mode
modal_base = genotypes.mode(axis=0)

# compute binarized matrix
binarized_genotypes = pd.DataFrame(index=one_thousand_genomes.index)

for i in range(3,10104):
    binarized_genotypes[i-3] = (genotypes[i] != modal_base.loc[0,i])

binarized_genotypes = binarized_genotypes.astype(int)

# subtract out mean (but do nothing with variance)
# note: don't think I actually need to do that, since sklearn automatically subtracts mean
# binarized_genotypes = binarized_genotypes - binarized_genotypes.mean()

# fit PCA
n_components = 3
pca = PCA(n_components=n_components)
transformed = pca.fit_transform(binarized_genotypes.values)

# plot
cmap1 = cm.get_cmap('jet', len(populations.unique()))

fig, ax = plt.subplots(figsize=(8,8))

for idx, p in enumerate(populations.unique()):
    to_plot = transformed[populations==p]
    ax.scatter(to_plot[:,0], to_plot[:,1], color=cmap1(idx), label='{} ({})'.format(p, pop_dict[p]))
    
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.legend()
plt.savefig('figs/miniproject4.part2.b.png')
plt.clf()

# plot
cmap2 = cm.get_cmap('jet', len(sexes.unique()))

fig, ax = plt.subplots(figsize=(8,8))

for idx, s in enumerate(sexes.unique()):
    to_plot = transformed[sexes==s]
    ax.scatter(to_plot[:,0], to_plot[:,2], color=cmap2(idx), label='{}'.format(sex_dict[s]))
    
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')

plt.legend()
plt.savefig('figs/miniproject4.part2.d.png')
plt.clf()

# plot
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(range(pca.components_.shape[1]), np.abs(pca.components_[2]), alpha=0.5)
ax.set_xlabel('Nucleobase index')
ax.set_ylabel('abs(PC3_i)')
plt.savefig('figs/miniproject4.part2.f.png')
plt.clf()


