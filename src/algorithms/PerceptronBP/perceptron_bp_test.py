from algorithms.PerceptronBP.PerceptronBPClassifier import PerceptronBPClassifier
from visual.confusion_matrix import printCorrectWrong


def test_perceptron_bp(trainingData, trainingLabels, testData, testLabels):
    clf = PerceptronBPClassifier()
    clf.fit(trainingData, trainingLabels)

    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
