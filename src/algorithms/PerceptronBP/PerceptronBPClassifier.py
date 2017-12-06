import numpy as np
from random import seed
import random

class PerceptronBPClassifier:
    def __init__(self, nEpoch, learningRate):
        self.weights = None
        self.trainingData = None
        self.trainingLabels = None
        self.nEpoch = nEpoch
        self.learningRate = learningRate

    def fit(self, trainingData, trainingLabels):
        self.trainingData = np.array(trainingData)
        self.trainingLabels = np.array(trainingLabels)
        seed(1)
        self.weights = [[random.uniform(-0.01, 0.01) for i in range(len(trainingData[0]) + 1)] for i in
                        range(len(np.unique(self.trainingLabels)))]
        # add bias
        self.trainingData = np.insert(self.trainingData, len(self.trainingData[0]), 1, axis=1)

        for epoch in range(self.nEpoch):
            for i, w in enumerate(self.weights):
                misplacedSamples = []
                misplacedLabels = []

                for j, data in enumerate(self.trainingData):
                    output = np.dot(w, data)

                    desiredClass = self.trainingLabels[j]
                    if output > 0 and i != desiredClass:
                        misplacedSamples.append(data)
                        misplacedLabels.append(-1)
                    if output < 0 and i == desiredClass:
                        misplacedSamples.append(data)
                        misplacedLabels.append(1)


                delta = [0 for i in range(len(data))]

                for k, d in enumerate(misplacedSamples):
                    delta = delta + misplacedLabels[k] * misplacedSamples[k]

                self.weights[i] = w + np.dot(self.learningRate, delta)

        return self.weights

    def predictSingle(self, sample):
        sample = np.insert(sample, len(sample), 1)
        sampleTimesWeight = np.dot(np.array(self.weights), sample)
        return np.argmax(sampleTimesWeight)

    def predict(self, data):
        predictions = []
        for sample in data:
            predictions.append(self.predictSingle(sample))
        return predictions
