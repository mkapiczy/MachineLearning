import warnings

import numpy as np
from sklearn.neighbors import NearestCentroid

from algorithms.MyNearestCentroid import MyNearestCentroid
from visual.confusion_matrix import createConfusionMatrix


def test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingImages)
        y = np.array(trainingLabels)
        clf = NearestCentroid()
        clf.fit(X, y)
        correct = 0
        wrong = 0
        for index, image in enumerate(testImages):
            prediction = clf.predict(image)
            if prediction == testLabels[index]:
                correct += 1
            else:
                wrong += 1

        print("Correct: " + str(correct))
        print("Wrong: " + str(wrong))


def test_nc_classify(trainingData, trainingLabels, testData, testLabels):
    clf = MyNearestCentroid()
    clf.fit(trainingData, trainingLabels)
    correct = 0
    wrong = 0
    for index, data in enumerate(testData):
        closestCentroidIndex = clf.predictSingle(data)
        if closestCentroidIndex == testLabels[index]:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))

    predictions = clf.predict(testData)
    createConfusionMatrix(predictions, testLabels)
