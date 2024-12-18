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
    centroids = rd.sample(data_points, num_clusters)
    cluster_assignments = [0] * len(data_points)

    for iteration in range(max_iter):
        for i, point in enumerate(data_points):
            distances = [distance(point, centroid) for centroid in centroids]
            cluster_assignments[i] = distances.index(min(distances))
        
        new_centroids = []
        for cluster_idx in range(num_clusters):
            cluster_points = [data_points[i] for i in range(len(data_points)) if cluster_assignments[i] == cluster_idx]
            if cluster_points:
                new_centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(rd.choice(data_points))
        centroid_shifts = [distance(new_centroids[i], centroids[i]) for i in range(num_clusters)]
        if max(centroid_shifts) < termination_tol:
            print(f"Coverged after {iteration + 1} iterations.")
            break
        centroids = new_centroids
    return centroids, cluster_assignments





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
