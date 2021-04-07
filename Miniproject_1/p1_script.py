import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(bins, filename = None):
	"""
	This function wraps a number of hairy matplotlib calls to smooth the plotting
	part of this assignment.

	Inputs:
	- bins: 	numpy array of shape max_bin_population X num_strategies numpy array. For this
				assignment this must be 200000 X 4.
				WATCH YOUR INDEXING! The element bins[i,j] represents the number of times the most
				populated bin has i+1 balls for strategy j+1.

	- filename: Optional argument, if set 'filename'.png will be saved to the current
				directory. THIS WILL OVERWRITE 'filename'.png
	"""
	assert bins.shape == (200000,4), "Input bins must be a numpy array of shape (max_bin_population, num_strategies)"
	assert np.array_equal(np.sum(bins, axis = 0),(np.array([30,30,30,30]))), "There must be 30 runs for each strategy"

	thresh =  max(np.nonzero(bins)[0])+3
	n_bins = thresh
	bins = bins[:thresh,:]
	print ("\nPLOTTING: Removed empty tail. Only the first non-zero bins will be plotted\n")

	ind = np.arange(n_bins)
	width = 1.0/6.0

	fig, ax = plt.subplots()
	rects_strat_1 = ax.bar(ind + width, bins[:,0], width, color='yellow')
	rects_strat_2 = ax.bar(ind + width*2, bins[:,1], width, color='orange')
	rects_strat_3 = ax.bar(ind + width*3, bins[:,2], width, color='red')
	rects_strat_4 = ax.bar(ind + width*4, bins[:,3], width, color='k')

	ax.set_ylabel('Number Occurrences in 30 Runs')
	ax.set_xlabel('Number of Balls In The Most Populated Bin')
	ax.set_title('Histogram: Load on Most Populated Bin For Each Strategy')

	ax.set_xticks(ind)
	ax.set_xticks(ind+width*3, minor = True)
	ax.set_xticklabels([str(i+1) for i in range(0,n_bins)], minor = True)
	ax.tick_params(axis=u'x', which=u'minor',length=0)

	ax.legend((rects_strat_1[0], rects_strat_2[0], rects_strat_3[0], rects_strat_4[0]), ('Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'))
	plt.setp(ax.get_xmajorticklabels(), visible=False)

	if filename is not None: plt.savefig(filename+'.png', bbox_inches='tight')

	plt.show()

# =======
# =======


#### Part 1

### (a)

## Strategy 1

def strat_1(N):
    """
    Input:
    - N: Integer number of bins and balls to be allocated

    Output:
    - bins: numpy array of dimensions 1 by N containing ball     placement.
    """

    bins = np.array([0] * N)

    for i in range(N):
    	pick = np.random.randint(0, N)
    	bins[pick] += 1

    return bins

# Strategy 2

def strat_2(N):
    """
    Input:
    - N: Integer number of bins and balls to be allocated

    Output:
    - bins: numpy array of dimensions 1 by N containing ball     placement.
    """

    bins = np.array([0] * N)

    N_picks = 2

    for i in range(N):
        picks = []
        picks_counts = []

        for n in range(N_picks):
            current_pick = np.random.randint(0, N)
    		picks.append(current_pick)
    		picks_counts.append(bins[current_pick])

    	bin_placement = picks[np.argmin(picks_counts)]
    	bins[bin_placement] += 1

    return bins

# Strategy 3
def strat_3(N):
	"""
	Input:
	- N: Integer number of bins and balls to be allocated

	Output:
	- bins: numpy array of dimensions 1 by N containing ball placement.

	"""

	N_picks = 3

	bins = np.array([0] * N)

	for i in range(N):
	    picks = []
	    picks_counts = []

	    for n in range(N_picks):
	        current_pick = np.random.randint(0, N)
	        picks.append(current_pick)
	        picks_counts.append(bins[current_pick])

	    bin_placement = picks[np.argmin(picks_counts)]
	    bins[bin_placement] += 1

	return bins

# Strategy 4

def strat_4(N):
	"""
	Input:
	- N: Integer number of bins and balls to be allocated

	Output:
	- bins: numpy array of dimensions 1 by N containing ball placement.

	"""

	bins = np.array([0] * N)

	for i in range(N):
		picks = []
		picks_counts = []

		first_pick = np.random.randint(0, N/2)
		second_pick = np.random.randint(N/2, N)
		picks.append(first_pick)
		picks.append(second_pick)
		picks_counts.append(bins[first_pick])
		picks_counts.append(bins[second_pick])

		if len(set(picks_counts)) != 1:
			bin_placement = picks[np.argmin(picks_counts)]
			bins[bin_placement] += 1
		else:
			bin_placement = picks[0]
			bins[bin_placement] += 1

	return bins

# (b)

repititions = 30
strat_1_maxes

for i in range(30):

strat_1_outcome = strat_1(200000)
strat_2_outcome = strat_2(200000)
strat_3_outcome = strat_3(200000)
strat_4_outcome = strat_4(200000)

all_dem = np.stack((strat_1_outcome, strat_2_outcome, strat_3_outcome, strat_4_outcome), axis = 0)

print(all_dem)

a = np.array([1]*200000)
b = np.array([3]*200000)
c = np.array([7]*200000)
d = np.array([4]*200000)

e = np.stack((a,b,c,d), axis = 0)
plot_histogram(c)
np.amax(strat_2(20000))



# Part 2
