import numpy as np
import scipy

from algorithms.DataClass import DataClass
from utils import chunks


class PerceptronMSEClassifier:
    def __init__(self):
        self.weights = None
        self.trainingData = None
        self.trainingLabels = None


    def sort_and_deduplicate(self, l):
        return list(np.unique(sorted(l, reverse=True)))


    def pseudo_inverse(self, matrix):
        print(str(matrix))
        mTranspose = matrix.transpose()
        print(str(matrix.transpose()))
        return np.linalg.inv(np.dot(matrix, mTranspose))


    def fit(self, trainingData, trainingLabels):
        self.trainingData = np.array(trainingData)
        self.trainingLabels = np.array(trainingLabels)

        if len(trainingData) > 10000:
            trainingDataInChunks = chunks(trainingData, 10000)
            labelsInChunks = chunks(trainingLabels, 10000)
            weights = []
            for i, data in enumerate(trainingDataInChunks):
                data = np.array(data)
                labels = np.array(labelsInChunks[i])
                pseudoInverse = np.linalg.pinv(data)
                labelsTranspose = labels.transpose()
                newLabels = []
                for label in labelsTranspose:
                    newLabels.append(self.to_zero_form(label, 10))
                weights.append(np.dot(pseudoInverse, newLabels))
            self.weights = np.mean(weights, axis=0, dtype=np.float64)
            print(str(self.weights))
        else:
            pseudoInverse = np.linalg.pinv(self.trainingData)
            labelsTranspose = self.trainingLabels.transpose()
            newLabels = []
            for label in labelsTranspose:
                newLabels.append(self.to_zero_form(label, 10))
            self.weights = np.dot(pseudoInverse, newLabels)



    def to_zero_form(self, number, numberOfClasses):
        array = []
        for i in range(numberOfClasses):
            if i == number:
                array.append(1)
            else:
                array.append(0)
        return array

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def predictSingle(self, sample):
        sampleTimesWeight = np.dot(sample, self.weights)
        return self.find_nearest(sampleTimesWeight, 1)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions