from algorithms.PerceptronBP.PerceptronBPClassifier import PerceptronBPClassifier
from visual.confusion_matrix import printCorrectWrong


def test_perceptron_bp(trainingData, trainingLabels, testData, testLabels):
    clf = PerceptronBPClassifier()
    clf.fit(trainingData, trainingLabels)
    #
    # prediction = clf.predictSingle(trainingData[0])
    # print("Prediction " + str(prediction) + " Label " + str(trainingLabels[0]))
    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
