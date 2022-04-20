# This script takes in the clusters found in 2 and re runs the descriptors for an image through it, so a histogram vector is generated
# that contains the amount that each cluster from the model occurs in the image
# Input: K-means models from 2 and descriptors from 1
# Output: A feature vector for each image that contains number of cluster occurences in the image. Also is repeated for k-means models of varying number of clusters

import pickle
import numpy as np

# perform fitting for each cluster model
for k in range(4, 35):
    # open the model
    with open('2_clusters_5/kmeans' + str(k), 'rb') as f:
        mod = pickle.load(f)

    print(mod)

    # go over each set of descriptors for every image
    vecs = np.zeros((857, k))
    for i in range(0, 857):
        # load descriptors
        with open('1_descriptors_5/des' + str(i), 'rb') as f:
            t = pickle.load(f)

        # get corresponding set of features for image
        features = mod.predict(t)

        # create histogram vector of each feature
        vec = np.zeros((1, k))
        for j in features:
            vec[0][j] += 1

        #print(vecs[0].shape)
        vecs[i] = vec

    # save results
    print(k, vecs.shape)
    with open('3_featureVectors_5/kmeans' + str(k), 'wb') as f:
        pickle.dump(vecs, f)
