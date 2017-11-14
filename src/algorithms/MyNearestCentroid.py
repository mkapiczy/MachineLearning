import numpy as np

from loader import MNIST



class MyNearestCentroid:
    def __init__(self):
        self.dataInClasses = []
        self.centroid = []
        self.trainingData = None
        self.trainingLabeles = None

    class Class:
        def __init__(self, label):
            self.data = []
            self.label = label


        def addData(self, data):
            self.data.append(data)

    class Centroid:
        def __init__(self, label):
            self.value = []
            self.label = label

    def sort_and_deduplicate(self, l):
        return list(np.unique(sorted(l, reverse=True)))

    def __devideIntoClasses(self, data, labels):
        uniqueLabels = self.sort_and_deduplicate(labels)
        classes = [self.Class(l) for l in uniqueLabels]
        for index, l in enumerate(labels):
            classes[l].label = l
            classes[l].addData(data[index])
        dataIntoClasses = np.array(classes)
        return dataIntoClasses

    def calculateCentroids(self):
        centroids = [self.Centroid(x.label) for x in self.dataInClasses]
        for index, singleClass in enumerate(self.dataInClasses):
            centroids[index].value = np.mean(np.matrix(singleClass.data), axis=0, dtype=np.float64)
        return centroids

    def closest(self, sample):
        aux = []
        for centroid in self.centroids:
            aux.append(abs(self.euclideanDistance(sample, centroid.value)))
        return self.centroids[aux.index(min(aux))].label

    def euclideanDistance(self, a, b):
        return np.linalg.norm(a - b, ord=2)

    def fit(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.dataInClasses = self.__devideIntoClasses(self.trainingData, self.trainingLabels)
        self.centroids = self.calculateCentroids()

    def predict(self, sample):
        sampleMeanValue = np.mean(np.matrix(sample), axis=0, dtype=np.float64)
        return self.closest(sampleMeanValue)