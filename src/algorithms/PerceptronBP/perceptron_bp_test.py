from algorithms.PerceptronBP.PerceptronBPClassifier import PerceptronBPClassifier
from visual.confusion_matrix import printCorrectWrong


def test_perceptron_bp(trainingData, trainingLabels, testData, testLabels, nEpoch, learningRate):
    clf = PerceptronBPClassifier(nEpoch, learningRate)
    clf.fit(trainingData, trainingLabels)

    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
