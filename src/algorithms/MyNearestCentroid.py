import numpy as np


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