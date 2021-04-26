import numpy as np
import matplotlib.pyplot as plt

# part (a) warm up
def pca_recover(X, Y):
    M = np.array([X,Y])
    u,s,vh = np.linalg.svd(M.T)
    return vh[0,1] / vh[0,0]

def ls_recover(X, Y):
    return (X @ Y.T) / np.linalg.norm(X)**2

#initialize data
X = np.arange(0.001, 1.001, 0.001)
Y = X*2

#mean-center the data
X -= X.mean()
Y -= Y.mean()

print("Warm-up: Slopes should be 2")
print("PCA slope: {}".format(pca_recover(X, Y)))
print("LS slope: {}".format(ls_recover(X,Y)))
print("\n")

# part (b) uniform distributions
# repeat trials 100 times to compute mean and variance:
ntrials = 1000
N = 1000 #sample size
pca_ms = []
ls_ms = []
for _ in range(ntrials):
    X = np.random.uniform(0, 1, size =(1,N))
    Y = np.random.uniform(0, 1, size = (1,N))
    #mean-center the data
    X -= X.mean()
    Y -= Y.mean()
    pca_ms.append(pca_recover(X,Y))
    ls_ms.append(ls_recover(X,Y))

print("Part (b): Uniform random points [0,1] x [0,1]")
print("Mean PCA slope: {:.2f}\tvariance: {:.3f}".format(np.mean(pca_ms), np.var(pca_ms)))
print("LS slope: {:.2f}\tvariance: {:.3f}".format(np.mean(ls_ms), np.var(ls_ms)))
print("\n")

# part (c) Noise in Y
nTrials = 30
noise_levels = np.arange(0, .505, 0.05)
X = np.arange(0.001, 1.001, 0.001)
X -= X.mean() #mean-center data
pca_results = []
ls_results = []
x_plot = []
for c in noise_levels:
    x_plot += [c] * nTrials
    for _ in range(nTrials):
        Y = 2*X + np.random.randn(len(X))*np.sqrt(c)
        Y -= Y.mean() #mean-center data
        pca_results.append(pca_recover(X, Y))
        ls_results.append(ls_recover(X, Y))
plt.figure(0)
plt.scatter(x_plot, pca_results, c = 'red')
plt.scatter(x_plot, ls_results, c = 'blue')
plt.title("Part 1(c): Noisy Y")
plt.savefig('figs/mp4_part1c.noisy_Y.png')

# part (d) Noise in X and Y
nTrials = 30
noise_levels = np.arange(0, .505, .05)
pca_results = []
ls_results = []
x_plot = []
for c in noise_levels:
    x_plot += [c] * nTrials
    for _ in range(nTrials):
        # initialize "clean" X
        X = np.arange(0, 1.001, .001)
        # create noisy Y from X
        Y = 2*X + np.random.randn(len(X)) * np.sqrt(c)
        # add noise to X
        X += np.random.randn(len(X)) * np.sqrt(c)

        # mean-center data
        X -= X.mean()
        Y -= Y.mean()
        pca_results.append(pca_recover(X, Y))
        ls_results.append(ls_recover(X, Y))
plt.figure(1)
plt.scatter(x_plot, pca_results, c = 'red')
plt.scatter(x_plot, ls_results, c = 'blue')
plt.title("Part 1(d): Noisy X and Noisy Y")
plt.savefig('figs/mp4_part1d.noisy_X.noisy_Y.png')
