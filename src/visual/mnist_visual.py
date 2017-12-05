import warnings

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from loader import MNIST

mndata = MNIST('../../samples/MNIST/')

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
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)

    pca = PCA(n_components=2).fit(testImages)
    pca_2d_test = pca.transform(testImages)
    for x in pca_2d_test:
        label = clf.predict(x)

        color, marker = getColorMarkerFromLabel(label)

        pyplot.scatter(x[0], x[1], c=color, marker=marker)

    # for centroid in clf.centroids:
    #     centroidValue = np.array(centroid.value)
    #
    #     centroidLabel = centroid.label
    #     pyplot.scatter(centroidValue[0][0], centroidValue[0][1], 60, c='#e50000', marker='x')
    #     pyplot.annotate(str(centroidLabel), centroidValue[0], centroidValue[0])

    zero_patch = mpatches.Patch(color='#15b01a', label='0')
    one_patch = mpatches.Patch(color='#0343df', label='1')
    two_patch = mpatches.Patch(color='#ff81c0', label='2')
    three_patch = mpatches.Patch(color='#653700', label='3')
    four_patch = mpatches.Patch(color='#cea2fd', label='4')
    five_patch = mpatches.Patch(color='#f97306', label='5')
    six_patch = mpatches.Patch(color='#ffff14', label='6')
    seven_patch = mpatches.Patch(color='#95d0fc', label='7')
    eight_patch = mpatches.Patch(color='#1fa774', label='8')
    nine_patch = mpatches.Patch(color='#ad8150', label='9')
    pyplot.legend(
        handles=[zero_patch, one_patch, two_patch, three_patch, four_patch, five_patch, six_patch, seven_patch,
                 eight_patch, nine_patch])
    pyplot.show()
