import random

from ORLData import ORLData


def preprocessData(data, labels):
    data = list(zip(*data))
    labelledData = []
    for i, d in enumerate(data):
        labelledData.append(ORLData(d, labels[i][0] - 1))

    def chunks(l, n):
        chunks = []
        for i in range(0, len(l), n):
            chunks.append(l[i:i + n])
        return chunks

    chunkedLabeledData = chunks(labelledData, 10)
    trainingDataLabeled = []
    testDataLabeled = []

    for d in chunkedLabeledData:
        x = 0
        train = 0
        tst = 0

        while x < 10:
            rand = random.choice(d);
            if train < 7:
                trainingDataLabeled.append(rand)
                train += 1
            else:
                testDataLabeled.append(rand)
                tst += 1
            x += 1

    trainingData = list(map(lambda z: z.data, trainingDataLabeled))
    trainingLabels = list(map(lambda z: z.label, trainingDataLabeled))
    testData = list(map(lambda z: z.data, testDataLabeled))
    testLabels = list(map(lambda z: z.label, testDataLabeled))

    return trainingData, trainingLabels, testData, testLabels