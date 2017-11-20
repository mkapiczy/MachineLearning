from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid

from algorithms.MyNearestCentroid import MyNearestCentroid
from loader import MNIST

from time import time

import numpy as np

from matplotlib import pyplot as plt

from sklearn import manifold, datasets
from matplotlib import pyplot
import warnings

mndata = MNIST('../samples/MNIST/')

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()
X = trainingImages
y = trainingLabels


def getColorMarkerFromLabel(label):
    if label == 0:
        color = '#15b01a'
        marker = 'p'
    elif label == 1:
        color = '#0343df'
        marker = 's'
    elif label == 2:
        color = '#ff81c0'
        marker = '8'
    elif label == 3:
        color = '#653700'
        marker = '<'
    elif label == 4:
        color = '#cea2fd'
        marker = '|'
    elif label == 5:
        color = '#f97306'
        marker = '_'
    elif label == 6:
        color = '#ffff14'
        marker = 'D'
    elif label == 7:
        color = '#95d0fc'
        marker = 'v'
    elif label == 8:
        color = '#1fa774'
        marker = 'h'
    elif label == 9:
        color = '#ad8150'
        marker = '>'
    return color, marker


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    np.random.seed(0)

    X = np.array(trainingImages)
    y = np.array(trainingLabels)

    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)

    X = np.array(pca_2d)
    y = np.array(trainingLabels)
    clf = MyNearestCentroid()
    clf.fit(X, y)

    pca = PCA(n_components=2).fit(testImages)
    pca_2d_test = pca.transform(testImages)
    for i,x in enumerate(pca_2d):
        label = trainingLabels[i]

        color, marker = getColorMarkerFromLabel(label)

        pyplot.scatter(x[0], x[1], c=color, marker=marker)

    for centroid in clf.centroids:
        centroidValue = np.array(centroid.value)

        centroidLabel = centroid.label
        pyplot.scatter(centroidValue[0][0], centroidValue[0][1], 60, c='#e50000', marker='x')
        pyplot.annotate(str(centroidLabel), centroidValue[0], centroidValue[0])

    pyplot.show()
