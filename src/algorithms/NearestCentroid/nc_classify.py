import warnings

import numpy as np
from sklearn.neighbors import NearestCentroid

from algorithms.NearestCentroid.MyNearestCentroid import MyNearestCentroid
from visual.confusion_matrix import createConfusionMatrix, printCorrectWrong


def test_nc_classify_with_sklearn(trainingData, trainingLabels, testData, testLabels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingData)
        y = np.array(trainingLabels)
        clf = NearestCentroid()
        clf.fit(X, y)

        predictions = clf.predict(testData)
        printCorrectWrong(predictions, testLabels)


def test_nc_classify(trainingData, trainingLabels, testData, testLabels):
    clf = MyNearestCentroid()
    clf.fit(trainingData, trainingLabels)
    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
