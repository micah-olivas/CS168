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
def total_distance(route): # get total distance of trip
    """
        Inputs:
            -route: a list of park names in trip order
        Output:
            -distance traveled along entire route
    """
    dist_trav = 0
    counter = 0
    prev_parks = []
    route_len = len(route)

    for p in route:
        if counter == 0:
            prev_parks.append(p)
            counter += 1
        else:
            if counter < (route_len - 1):
                dist_trav += edge_distance(long_lat(p), long_lat(prev_parks[-1]))
                counter += 1
                prev_parks.append(p)
            else:
                dist_trav += edge_distance(long_lat(p), long_lat(prev_parks[0]))
    return dist_trav

# demonstrate distance function works
distance(long_lat('Arches'), long_lat('Acadia'))

iter = 10
for i in range(iter):
    best_order = random_walk(dat)
    plot_trip(best_order)

# define MCMC
def switch_two(route):
    new = np.copy(route)
    ix1 = np.random.randint(len(new))
    ix2 = ix1 + np.random.choice([-1,1])
    if ix2 == 31:
        ix2 = 0

    new[ix1], new[ix2] = new[ix2], new[ix1]
    return new

iter = 1000

def maxiter(dat, T, iter):
    route = random_walk(dat)
    best_route = np.copy(route)

    for i in range(iter):
        new_route = switch_two(route)
        old_route_dist = total_distance(route)
        new_route_dist = total_distance(new_route)
        dist_best = total_distance(best_route)

        dist_change = new_route_dist - old_route_dist
        if (dist_change < 0) or (T > 0 and (np.random.uniform(0,1) < np.exp(-(dist_change/T))):
            route = np.copy(new_route)
        if new_route_dist < dist_best:
            best_route = np.copy(new_route)

        plot_trip(best_route)

    print(dist_best)

# part 2b
temps = [0, 1, 10, 1000]
MAXITER = 100

for t in temps:
    maxiter(dat, 1, MAXITER)




route
