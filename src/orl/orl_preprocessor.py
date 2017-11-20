import random

from orl.ORLData import ORLData
from utils import chunks


def preprocessData(data, labels):
    data = list(zip(*data))
    labelledData = []
    for i, d in enumerate(data):
        labelledData.append(ORLData(d, labels[i][0] - 1))

    dataInClasses = chunks(labelledData, 10)
    trainingDataLabeled = []
    testDataLabeled = []

    for dataClass in dataInClasses:
        x = 0
        train = 0
        tst = 0

        while x < 10:
            randomSample = random.choice(dataClass)
            if train < 7:
                trainingDataLabeled.append(randomSample)
                train += 1
            else:
                testDataLabeled.append(randomSample)
                tst += 1
            x += 1

    trainingData = list(map(lambda z: z.data, trainingDataLabeled))
    trainingLabels = list(map(lambda z: z.label, trainingDataLabeled))
    testData = list(map(lambda z: z.data, testDataLabeled))
    testLabels = list(map(lambda z: z.label, testDataLabeled))

    return trainingData, trainingLabels, testData, testLabels