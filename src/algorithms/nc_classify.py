import warnings
import numpy as np
from sklearn.neighbors import NearestCentroid


class MyNearestCentroid:
    def __init__(self):
        self.dataInClasses = []
        self.centroid = []
        self.trainingData = None
        self.trainingLabeles = None

    def __uniq(self, lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item

    def __sort_and_deduplicate(self, l):
        return list(self.__uniq(sorted(l, reverse=True)))

    def __devideIntoClasses(self, data, labels):
        numberOfClasses = len(self.__sort_and_deduplicate(labels))
        dataIntoClasses = [[] for x in range(numberOfClasses)]
        for index, l in enumerate(labels):
            dataIntoClasses[l].append(data[index])
        dataIntoClasses = np.array(dataIntoClasses)
        return dataIntoClasses

    def __calculateCentroids(self, dataDividedInClasses):
        centroids = [0 for x in range(len(dataDividedInClasses))]
        for index, singleClassData in enumerate(dataDividedInClasses):
            centroids[index] = np.mean(np.matrix(singleClassData), axis=0, dtype=np.float64)
        return centroids

    def __closest(self, list, elem):
        aux = []
        for valor in list:
            aux.append(abs(self.__euclideanDistance(elem, valor)))
        return aux.index(min(aux))

    def __euclideanDistance(self, a, b):
        return np.linalg.norm(a - b, ord=2)

    def fit(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.dataInClasses = self.__devideIntoClasses(self.trainingData, self.trainingLabels)
        self.centroids = self.__calculateCentroids(self.dataInClasses)

    def predict(self, sample):
        sampleMeanValue = np.mean(np.matrix(sample), axis=0, dtype=np.float64)
        return self.__closest(self.centroids, sampleMeanValue)


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


def test_nc_classify(trainingData, trainingLabels, testData, testLabels):
    clf = MyNearestCentroid()
    clf.fit(trainingData, trainingLabels)

    correct = 0
    wrong = 0
    for index, data in enumerate(testData):
        closestCentroidIndex = clf.predict(data)
        if closestCentroidIndex == testLabels[index]:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))
