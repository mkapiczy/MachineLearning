from algorithms.PerceptronMSE.PerceptronMSEClassifier import PerceptronMSEClassifier
from visual.confusion_matrix import printCorrectWrong


def test_perceptron_mse(trainingData, trainingLabels, testData, testLabels):
    clf = PerceptronMSEClassifier()
    clf.fit(trainingData, trainingLabels)

    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
