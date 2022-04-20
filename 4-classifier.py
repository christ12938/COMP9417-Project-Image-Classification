# This script takes in the feature vectors from 3 and performs a classification model on them to predict the class they belong to
# Input: Feature vectors produced in 3, and access to the training classes to make the y vector
# Output: A trainined model and their respective f1 scores

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from os.path import exists
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# load in y values based on the folder that the image is from
y = np.zeros(858)
for i in range (0, 858):
    val = 0
    if exists("MLCancer/train/1/" + str(i) + ".png"):
        val = 1
    elif exists("MLCancer/train/2/" + str(i) + ".png"):
        val = 2
    elif exists("MLCancer/train/3/" + str(i) + ".png"):
        val = 3
    y[i] = val

print(y.shape, y)


# A portition of this work is referenced from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# create names list
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
]

# create model list that initialises each model
models = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
]

#  open the feature vector to model
with open('3_featureVectors_5/kmeans32', 'rb') as f:
    X = pickle.load(f)

# standise the data
X = StandardScaler().fit_transform(X)
print(X.shape)


f1s = [[], [], [], [], [], [], [], [], []]
for j, mod in enumerate(models):
    ave = []
    # loop to so large number of random states are used, so f1 value is closer to real
    for x in range(0, 50):
        # split data on random state
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=x)
        # fit and predict the model
        y_pred = mod.fit(X_train, y_train).predict(X_test)
        # calculate the f1 score
        f1 = f1_score(y_test, y_pred, average='macro')      
        ave.append(f1)

    # calculate average and store
    f1s[j] = sum(ave) / len(ave)
    print(names[j] + " f1 score:", f1s[j])