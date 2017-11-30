from algorithms.CrossValidator import cross_val_score
from algorithms.MyNearestSubclassCentroid import MyNearestSubclassCentroid

import numpy as np

from visual.confusion_matrix import createConfusionMatrix


def validateHyperParameter(trainingData, trainingLabels, numberOfSubclasses):
    clf = MyNearestSubclassCentroid(numberOfSubclasses)
    scores = cross_val_score(clf, trainingData, np.array(trainingLabels), cv=5)
    print(str(scores))


def test_nsc_classify(trainingData, trainingLabels, testData, testLabels, numberOfSubclasses):
    clf = MyNearestSubclassCentroid(numberOfSubclasses)
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
