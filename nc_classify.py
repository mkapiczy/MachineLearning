import warnings

import numpy as np
from sklearn.neighbors import NearestCentroid

from loader import MNIST

mndata = MNIST('./samples/MNIST/')

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def devideIntoClasses(data, labels):
    numberOfClasses = len(sort_and_deduplicate(labels))
    dataIntoClasses = [[] for x in range(numberOfClasses)]
    for index, l in enumerate(labels):
        dataIntoClasses[l].append(data[index])
    dataIntoClasses = np.array(dataIntoClasses)
    return dataIntoClasses


def calculateCentroids(dataDividedInClasses):
    centroids = [0 for x in range(len(dataDividedInClasses))]
    for index, singleClassData in enumerate(dataDividedInClasses):
        centroids[index] = np.mean(np.matrix(singleClassData), axis=0, dtype=np.float64)
    return centroids


def closest(list, elem):
    aux = []
    for valor in list:
        aux.append(abs(euclideanDistance(elem, valor)))
    return aux.index(min(aux))


def euclideanDistance(a, b):
    return np.linalg.norm(a - b, ord=2)

def nc_classify(trainData, testData, testLabels, trainLabels):
    dataInClasses = devideIntoClasses(trainData, trainLabels)
    centroids = calculateCentroids(dataInClasses)
    test_nc_classify(testData, testLabels, centroids)

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

def test_nc_classify(testData, testLabels, centroids):
    correct = 0
    wrong = 0

    for index, data in enumerate(testData):
        dataMeanValue = np.mean(np.matrix(data), axis=0, dtype=np.float64)
        closestCentroidIndex = closest(centroids, dataMeanValue)
        if closestCentroidIndex == testLabels[index]:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))