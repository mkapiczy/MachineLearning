from loader import MNIST
from nc_classify import nc_classify
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

mndata = MNIST('./samples/MNIST/')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()


def test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels):
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


nc_classify(trainingImages, testImages, testLabels, trainingLabels)
test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)