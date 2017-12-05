import numpy as np

from utils import transformLabels


class PerceptronMSEClassifier:

    def __init__(self):
        self.weights = None
        self.trainingData = None
        self.trainingLabels = None

    def fit(self, trainingData, trainingLabels):
        self.trainingData = np.array(trainingData)
        self.trainingLabels = np.array(trainingLabels)
        # add bias
        self.trainingData = np.insert(self.trainingData, len(self.trainingData[0]), 1, axis=1)

        pseudoInverse = np.linalg.pinv(self.trainingData)
        labels = transformLabels(self.trainingLabels)

        self.weights = np.dot(pseudoInverse, labels)

    def findIndexOfNearestElemToValue(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def predictSingle(self, sample):
        sample = np.insert(sample, len(sample), 1)
        predictionVector = np.dot(self.weights.transpose(), sample)
        return self.findIndexOfNearestElemToValue(predictionVector, 1)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions
