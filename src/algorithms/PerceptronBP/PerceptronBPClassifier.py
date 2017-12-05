import numpy as np

from utils import transformLabels
from random import seed
import random


class PerceptronBPClassifier:
    def __init__(self):
        self.weights = None
        self.nEpoch = 5
        self.learningRate = 0.005

    def fit(self, trainingData, trainingLabels):
        self.trainingData = np.array(trainingData)
        self.trainingLabels = np.array(trainingLabels)
        seed(1)
        labels = transformLabels(self.trainingLabels)
        self.weights = [[0 for i in range(len(trainingData[0]) + 1)] for i in
                        range(len(np.unique(self.trainingLabels)))]
        # add bias
        self.trainingData = np.insert(self.trainingData, len(self.trainingData[0]), 1, axis=1)

        for epoch in range(self.nEpoch):
            for i, data in enumerate(self.trainingData):
                prediction = []
                for w in self.weights:
                    p = np.dot(np.array(w).transpose(), data)
                    prediction.append(p)
                predictedLabel = np.argmax(prediction)
                # print("Predicted : " + str(predictedLabel) + " - " + str(self.trainingLabels[i]))
                # print(str(prediction))
                if predictedLabel != self.trainingLabels[i]:
                    error = np.array(labels[i]) - np.array(prediction)
                    # print(str(error))
                    self.weights[self.trainingLabels[i]] = self.weights[self.trainingLabels[i]] + self.learningRate * data * -1

        return self.weights

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def predictSingle(self, sample):
        sample = np.insert(sample, len(sample), 1)
        sampleTimesWeight = np.dot(self.weights, sample)
        return self.find_nearest(sampleTimesWeight, 1)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions
