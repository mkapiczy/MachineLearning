from algorithms.CrossValidator import cross_val_score
from algorithms.PerceptronMSE.PerceptronMSEClassifier import PerceptronMSEClassifier
from visual.confusion_matrix import createConfusionMatrix
import numpy as np

def validateHyperParameter(trainingData, trainingLabels):
    clf = PerceptronMSEClassifier()
    scores = cross_val_score(clf, trainingData, np.array(trainingLabels), cv=5)
    print(str(scores))

def test_perceptron_mse(trainingData, trainingLabels, testData, testLabels):
    clf = PerceptronMSEClassifier()
    clf.fit(trainingData, trainingLabels)

    correct = 0
    wrong = 0
    for index, data in enumerate(testData):
        prediction = clf.predictSingle(data)
        if prediction == testLabels[index]:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))

    # predictions = clf.predict(testData)
    # createConfusionMatrix(predictions, testLabels)
