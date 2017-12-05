import math
import numpy as np
import scipy

from algorithms.DataClass import DataClass
from utils import chunks


class PerceptronBPClassifier:
    def __init__(self):
        self.weights = None
        self.nEpoch = 5
        self.learningRate = 0.05

    def sort_and_deduplicate(self, l):
        return list(np.unique(sorted(l, reverse=True)))

    def pseudo_inverse(self, matrix):
        print(str(matrix))
        mTranspose = matrix.transpose()
        print(str(matrix.transpose()))
        return np.linalg.inv(np.dot(matrix, mTranspose))

    def transformLabels(self, labels):
        labelsTranspose = labels.transpose()
        newLabels = []
        for label in labelsTranspose:
            newLabels.append(self.to_zero_form(label, 10))
        return newLabels

    def fit(self, trainingData, trainingLabels):
        self.trainingData = np.array(trainingData)
        self.trainingLabels = np.array(trainingLabels)


        labels = self.transformLabels(self.trainingLabels)
        pseudoInverse = np.linalg.pinv(self.trainingData)
        self.weights = np.dot(pseudoInverse, labels)

        # print(str(len(labels))) 60000
        # print(str(len(self.trainingData))) 60000
        for epoch in range(self.nEpoch):
            for i, data in enumerate(self.trainingData):
                prediction = np.dot(self.weights.transpose(), data)
                # print(str(labels[i]))
                error = []
                for pred in prediction:
                    print(str(pred))
                    error.append(labels[i]-pred)
                # print(str(error))
                error = np.mean(error, axis=0, dtype=np.float64)
                # print(str(error))

                self.weights = self.weights + self.learningRate * error
        return self.weights

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
