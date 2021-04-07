import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Heatmap
def makeHeatMap(data, names, color, outputFileName):
	#to catch "falling back to Agg" warning
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
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

		ax.set_xticklabels(range(1, 21))
		ax.set_yticklabels(names)

		plt.tight_layout()

		plt.savefig(outputFileName, format = 'png')
		plt.close()

### part 1
try:
    wow = open('data50.csv')
    print(wow)

finally:
    wow.close()
### part 2

### part 3

## (a)

# similarity search with cosine similarity nn classification
(sum(x*y)/

array = np.array([[1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [0,1], ])
np.shape(array)

# compute matrix of (A,B)
np.matrix(array)
