import numpy as np

from algorithms.CrossValidator import cross_val_score
from algorithms.PerceptronBP.PerceptronBPClassifier import PerceptronBPClassifier
from algorithms.PerceptronMSE.PerceptronMSEClassifier import PerceptronMSEClassifier


def validateHyperParameter(trainingData, trainingLabels):
    clf = PerceptronMSEClassifier()
    scores = cross_val_score(clf, trainingData, np.array(trainingLabels), cv=5)
    print(str(scores))

def test_perceptron_bp(trainingData, trainingLabels, testData, testLabels):
    clf = PerceptronBPClassifier()
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

    # predictions = clf.predict(testData)
    # createConfusionMatrix(predictions, testLabels)
