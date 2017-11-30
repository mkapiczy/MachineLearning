import numpy as np
import warnings

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from algorithms.DataClass import DataClass
from algorithms.MyNearestCentroid import MyNearestCentroid
from loader import MNIST


class MyNearestSubclassCentroid(MyNearestCentroid):
    def __init__(self, numberOfSubclasses):
        self.dataInClasses = []
        self.centroids = []
        self.trainingData = None
        self.trainingLabels = None
        self.numberOfSubclasses = numberOfSubclasses

    def __partition(self, lst, n):
        division = len(lst) / n
        return np.asarray([lst[round(division * i):round(division * (i + 1))] for i in range(n)])


    def fartest(self, cluster):
        aux = []
        for point in cluster.data:
            aux.append(abs(self.euclideanDistance(point, np.mean(np.matrix(cluster.data), axis=0, dtype=np.float64))))
        return cluster.data[aux.index(max(aux))]

    def dealWithEmptyClusters(self, subclasses):
        for subclass in subclasses:
            if len(subclass.data) == 0:
                maxCluster = None
                maxClusterIndex = 0
                for index, cl in enumerate(subclasses):
                    if maxCluster == None or len(cl.data) > len(maxCluster.data):
                        maxCluster = cl
                        maxClusterIndex = index

                fartestPoint = self.fartest(subclasses[maxClusterIndex])
                subclasses[maxClusterIndex].data.remove(fartestPoint)
                subclass.addData(fartestPoint)
        return subclasses



    def __devideIntoClasses(self, data, labels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uniqueLabels = self.sort_and_deduplicate(labels)
            classes = [DataClass(l) for l in uniqueLabels]
            for index, l in enumerate(labels):
                classes[l].label = l
                classes[l].addData(data[index])

            newClasses = []
            for singleClass in classes:

                pca = PCA(n_components=2).fit(singleClass.data)
                pca_2d = pca.transform(singleClass.data)
                # print(str(len(pca_2d)))
                kmeans = KMeans(self.numberOfSubclasses)
                kmeans.fit(np.array(pca_2d))

                subclasses = [DataClass(singleClass.label) for x in range(self.numberOfSubclasses)]

                for i, data in enumerate(pca_2d):
                    prediction = kmeans.predict(data)[0]
                    subclasses[prediction].addData(singleClass.data[i])

                subclasses = self.dealWithEmptyClusters(subclasses)

                for subclass in subclasses:
                    newClasses.append(subclass)

            return np.array(newClasses)



    def fit(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.dataInClasses = self.__devideIntoClasses(self.trainingData, self.trainingLabels)
        self.centroids = self.calculateCentroids()

    def predictSingle(self, sample):
        sampleMeanValue = np.mean(np.matrix(sample), axis=0, dtype=np.float64)
        return self.closest(sampleMeanValue)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions
