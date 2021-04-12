import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part_1 import *

### part 3

# Locality-Sensitive-Hashing

# Hyperplane Hashing
mu, sigma = 0, 1 # initialize mean and sd
s = np.random.normal(mu, sigma, 1000)

# Classification
