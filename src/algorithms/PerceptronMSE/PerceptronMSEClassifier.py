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
                array.append(-1)
        return array

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def predictSingle(self, sample):
        # weight transpose times sample
        sampleTimesWeight = np.dot(self.weights.transpose(), sample)
        return self.find_nearest(sampleTimesWeight, 1)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions