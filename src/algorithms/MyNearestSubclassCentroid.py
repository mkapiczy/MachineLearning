import numpy as np

from algorithms.MyNearestCentroid import MyNearestCentroid


class MyNearestSubclassCentroid(MyNearestCentroid):
    def __init__(self, numberOfSubclasses):
        self.dataInClasses = []
        self.dataInSubclasses = []
        self.centroid = []
        self.trainingData = None
        self.trainingLabeles = None
        self.numberOfSubclasses = numberOfSubclasses

    def __partition(self, lst, n):
        division = len(lst) / n
        return np.asarray([lst[round(division * i):round(division * (i + 1))] for i in range(n)])

    def __devideIntoClasses(self, data, labels):
        uniqueLabels = self.sort_and_deduplicate(labels)
        classes = [self.Class(l) for l in uniqueLabels]
        for index, l in enumerate(labels):
            classes[l].label = l
            classes[l].addData(data[index])

        newClasses = []
        for singleClass in classes:
            np.random.shuffle(singleClass.data)
            splittedData = np.array_split(np.array(singleClass.data), self.numberOfSubclasses)
            for subClassData in splittedData:
                cl = self.Class(singleClass.label)
                for data in subClassData:
                    cl.addData(data)
                newClasses.append(cl)
        dataIntoClasses = np.array(newClasses)

        return dataIntoClasses

    def fit(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.dataInClasses = self.__devideIntoClasses(self.trainingData, self.trainingLabels)
        self.centroids = self.calculateCentroids()

    def predict(self, sample):
        sampleMeanValue = np.mean(np.matrix(sample), axis=0, dtype=np.float64)
        return self.closest(sampleMeanValue)
