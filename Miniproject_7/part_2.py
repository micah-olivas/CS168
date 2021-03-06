import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dat = pd.read_csv('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_7/parks.csv')

def long_lat(park): # get longitude and latitude from park name
    return np.array(dat.loc[dat['Name'] == park]).squeeze()[1:3]
def random_walk(dat): # select random walk across 30 parks
    route = np.array(dat['Name'])

    np.random.shuffle(route)
    start = route[0]
    route_loop = np.append(route, start)
    return route_loop
def plot_trip(trip_order): # plot directed graph of trip
    prior_long = 0
    prior_lat = 0

    for d in trip_order:
        row = dat.loc[dat['Name'] == d]
        long = float(row['Longitude'])
        latit = float(row['Latitude'])

        if prior_lat == 0:
            prior_long = long
            prior_lat = latit

        else:
            plt.arrow(x = long, y = latit, dx = (prior_long - long), dy = (prior_lat - latit), alpha=0.1, head_width=1, length_includes_head=True)
            prior_long = long
            prior_lat = latit
    return plt.scatter(x = dat['Longitude'], y = dat['Latitude'])
def edge_distance(long_lat_1_array, long_lat_2_array): # get length of single trip edge
    long_1 = long_lat_1_array[0]
    lat_1 = long_lat_1_array[1]
    long_2 = long_lat_2_array[0]
    lat_2 = long_lat_2_array[1]
    return np.sqrt((long_1 - long_2)**2 + (lat_1 - lat_2)**2)
def total_distance(route):
    coords = np.array([long_lat(i) for i in route]).squeeze()
    coord_roll = np.roll(coords, 1, axis=0)
    return np.sum(np.sqrt(list(np.sum(((coords - coord_roll)**2), axis=1)))) # distance by VECTORS

# define MCMC scheme
# part 2b and 2c
temps = [0, 1, 10, 1000]
trials = 10
MAXITER = 10000

# define MCMC, switching two stops at every iteration
def switch_any_two(route):
    new = route[:]
    ix1 = np.random.randint(len(new))
    ix2 = np.random.randint(len(new))
    new[ix1], new[ix2] = new[ix2], new[ix1]
    return new

def switch_adj_two(route):
    new = np.copy(route)
    ix1 = np.random.randint(len(new))
    ix2 = ix1 + np.random.choice([-1,1])
    if ix2 == 31:
        ix2 = 0
    new[ix1], new[ix2] = new[ix2], new[ix1]
    return new

def iterate_adj(dat, T, iter):
    route = np.ndarray.tolist(random_walk(dat))
    best_route = route[:]
    route_distances = [] #store distances for plotting

    for i in range(iter):
        new_route = switch_adj_two(route)
        old_route_dist = total_distance(route)
        new_route_dist = total_distance(new_route)
        route_distances.append(new_route_dist)
        dist_best = total_distance(best_route)

        dist_change = new_route_dist - old_route_dist
        if (dist_change < 0) or (T > 0 and (np.random.uniform(0,1) < np.exp(-dist_change/T))):
            route = new_route[:]
        if new_route_dist < dist_best:
            best_route = new_route[:]

    return dist_best, route_distances
def iterate_any(dat, T, iter):
    route = np.ndarray.tolist(random_walk(dat))
    best_route = route[:]
    route_distances = [] #store distances for plotting

    for i in range(iter):
        new_route = switch_any_two(route)
        old_route_dist = total_distance(route)
        new_route_dist = total_distance(new_route)
        route_distances.append(new_route_dist)
        dist_best = total_distance(best_route)

        dist_change = new_route_dist - old_route_dist
        if (dist_change < 0) or (T > 0 and (np.random.uniform(0,1) < np.exp(-dist_change/T))):
            route = new_route[:]
        if new_route_dist < dist_best:
            best_route = new_route[:]

    return dist_best, route_distances

def plot_switches_adj(dat, MAXITER=MAXITER, trials=trials, temps=temps):
    idxs = [(0,0), (0,1), (1,0), (1,1)]
    temp_idx = list(zip(temps, idxs))

    fig, axs = plt.subplots(2, 2)
    for ind in temp_idx:
        for i in range(trials):
            best, dist_iters = iterate_adj(dat, ind[0], MAXITER)
            axs[ind[1]].plot(range(MAXITER), dist_iters, )
            axs[ind[1]].set_title('temp '+str(ind[0]))

    for ax in axs.flat:
        ax.set(xlabel='MCMC Iteration', ylabel='Trip Distance')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.set_figheight(6)
    fig.set_figwidth(7)
    fig.tight_layout()
    plt.rcParams['lines.linewidth'] = 0.5
    fig.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_7/proj7_part2b')

def plot_switches_any(dat, MAXITER=MAXITER, trials=trials, temps=temps):
    idxs = [(0,0), (0,1), (1,0), (1,1)]
    temp_idx = list(zip(temps, idxs))

    fig, axs = plt.subplots(2, 2)
    for ind in temp_idx:
        for i in range(trials):
            best, dist_iters = iterate_any(dat, ind[0], MAXITER)
            axs[ind[1]].plot(range(MAXITER), dist_iters, )
            axs[ind[1]].set_title('temp '+str(ind[0]))

    for ax in axs.flat:
        ax.set(xlabel='MCMC Iteration', ylabel='Trip Distance')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.set_figheight(6)
    fig.set_figwidth(7)
    fig.tight_layout()
    plt.rcParams['lines.linewidth'] = 0.5
    fig.savefig('/Users/micaholivas/Desktop/Coursework/Algorithms_CS_168/Miniproject_7/proj7_part2c')

plot_switches_adj(dat)
plot_switches_any(dat)
