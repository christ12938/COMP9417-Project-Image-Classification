# This script takes in every image in the dataset, calculates the SIFT descriptors for it, and then uses Isomap to compress the dimenionality from 128 to a specified value
# Input: Dataset of images
# Output: List of descriptors for every image


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
import cv2
import pickle
from sklearn.decomposition import PCA


n = 858
ims = []

# load all images
for i in range(0, n):
    ims.append(cv2.imread('MLCancer/train/temp/' + str(i) + '.png'))

# plot test image
plt.figure(figsize=(5, 5), dpi=80)
imgplot = plt.imshow(cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB))



des = np.zeros((1, 5))

# iterate over each image to calculate data
for i, img in enumerate(ims):
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create the sift object, specifying limit of features
    sift = cv2.SIFT_create(nfeatures=100)
    
    # find the features
    kp, de = sift.detectAndCompute(gray,None)
    print(i, de.shape)

    # plotting the PCA variance
    # pca = PCA().fit(de)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')


    # perform dimension reduction with set amount of components
    embed = Isomap(n_components=10)
    trans = embed.fit_transform(de)

    # saving the data
    with open('1_SIFTDescriptors_100/des' + str(i), 'wb') as f:
        pickle.dump(trans, f)

    
    

    
# visualise the keypoints on the image
siftImg = cv2.drawKeypoints(gray,kp, img)
plt.figure(figsize=(5, 5), dpi=80)
imgplot = plt.imshow(siftImg)