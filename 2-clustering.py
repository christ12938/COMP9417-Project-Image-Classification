# This script takes in every descriptor from 1 and processes a clustering algorithm with k-means.
# Input: Every descriptor produced from 1
# Output: A number of trained k-means models based on varying the number of clusters 

import numpy as np
from sklearn.cluster import KMeans
import pickle


# load in all descriptors found in 1
x = np.zeros((1, 5))
for i in range(0, 857):

    with open('temp/des' + str(i), 'rb') as f:
        t = pickle.load(f)

    t = np.delete(t, (0), axis=0)
    print(i, t.shape)

    x = np.concatenate((x, t))

x = np.delete(x, (0), axis=0)
print('final: ', x.shape)



# performing clutering with varying number of cluster
for i in range(13, 35):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(x) 

    # save the result
    with open('clusters/kmeans' + str(i), 'wb') as f:
        pickle.dump(kmeans, f)



# print info
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[100, 0, 70, 110, 100]]))