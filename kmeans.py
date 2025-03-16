#!/usr/bin/env python
# coding: utf-8



# Some required imports.
# Make sure you have these packages installed on your system.
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rd




# Distance function used by the kmeans algorithm (euklidean distance)
def distance(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2))
    


# This method contains the implementation of the kmeans algorithm, which 
# takes following input parameters:
#   data_points - A Pandas DataFrame, where each row contains an x and y coordinate.
#   termination_tol - Terminate when the total distance (sum of distance between each datapoint
#                     and its centroid) is smaller than termination_tol. 
#   max_iter - Stop after maximum max_iter iterations.
#
# The method should return the following three items
#   centroids - Pandas Dataframe containing centroids
#   data_points - The data_points with added cluster column
#   total_dist - The total distance for the final clustering
#
# return centroids, data_points, total_dist
#
#  
def kmeans(data_points, num_clusters, termination_tol, max_iter):
    data_points = pd.DataFrame(data_points_list, columns=['x', 'y'])
    centroids = data_points.sample(n=num_clusters, random_state=None).reset_index(drop=True)

    data_points = data_points.copy()
    data_points['cluster'] = -1

    old_total_dist = float('inf')

    for _ in range(max_iter):
        for idx, point in data_points.iterrows():
            distances = []
            for c_idx in range(num_clusters):
                dist = distance((point['x'], point['y']),
                                (centroids.loc[c_idx, 'x'], centroids.loc[c_idx, 'y']))
                distances.append(dist)
            data_points.loc[idx, 'cluster'] = np.argmin(distances)

        new_centroids = []
        for c_idx in range(num_clusters):
            cluster_points = data_points[data_points['cluster'] == c_idx]
            if len(cluster_points) > 0:
                mean_x = cluster_points['x'].mean()
                mean_y = cluster_points['y'].mean()
            else:
                random_point = data_points.sample(n=1)
                mean_x = random_point['x'].values[0]
                mean_y = random_point['y'].values[0]
            new_centroids.append([mean_x, mean_y])

        new_centroids_df = pd.DataFrame(new_centroids, columns=['x','y'])

        total_dist = 0.0
        for idx, point in data_points.iterrows():
            c_idx = int(point['cluster'])
            centroid = new_centroids_df.loc[c_idx]
            total_dist += distance((point['x'], point['y']), (centroid['x'], centroid['y']))

        if abs(old_total_dist - total_dist) < termination_tol:
            centroids = new_centroids_df
            break

        centroids = new_centroids_df
        old_total_dist = total_dist

    return centroids, data_points, total_dist





# Test elbow method using this code

# Read data points from csv file
data_points = pd.read_csv("cluster_data_points.csv")

# Set termination criteria
termination_tol = 0.001
max_iter = 100


# Plot random data using matplotlib
fig, ax = plt.subplots()
ax.scatter(data_points['x'], data_points['y'], c='black')
plt.title("Data points")
# plt.show()


num_clusters_to_test = 15
total_dist_elbow = []

for k in range(1,num_clusters_to_test+1):
    data_points_list = data_points[['x', 'y']].values.tolist()
    kmeans_output = kmeans(data_points_list, k, termination_tol, max_iter)
    total_dist_elbow.append(kmeans_output[2])
    
#Plot elbow curve
plt.plot(list(range(1,num_clusters_to_test+1)), total_dist_elbow)
plt.title("Elbow method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Total distance")
plt.show()





# Plot clusters for different values of k using this code
data_points = pd.read_csv("cluster_data_points.csv")

termination_tol = 0.001
max_iter = 100

for k in range(1,11):
    

    kmeans_output = kmeans(data_points, k, termination_tol, max_iter)
    
    fig, ax = plt.subplots()
    ax.scatter(kmeans_output[0]['x'], kmeans_output[0]['y'], c='black', marker='*')

    for centroid_id in range(k):
        points = data_points.loc[kmeans_output[1]['cluster'] == centroid_id]

        ax.scatter(points['x'], points['y'])

    plt.title("Clusters for k=" + str(k))
    plt.show()
