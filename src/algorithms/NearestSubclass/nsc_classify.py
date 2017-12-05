from algorithms.NearestSubclass.MyNearestSubclassCentroid import MyNearestSubclassCentroid

from visual.confusion_matrix import printCorrectWrong, createConfusionMatrix


def test_nsc_classify(trainingData, trainingLabels, testData, testLabels, numberOfSubclasses):
    clf = MyNearestSubclassCentroid(numberOfSubclasses)
    clf.fit(trainingData, trainingLabels)

    predictions = clf.predict(testData)
    printCorrectWrong(predictions, testLabels)
    # createConfusionMatrix(predictions, testLabels)
