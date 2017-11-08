import random

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors.nearest_centroid import NearestCentroid

from ORLData import ORLData
from nc_classify import nc_classify

data = loadmat('./samples/ORL/orl_data.mat')['data']
labels = loadmat('./samples/ORL/orl_lbls.mat')['lbls']

data = list(zip(*data))
labelledData = []
for i, d in enumerate(data):
    labelledData.append(ORLData(d, labels[i][0]-1))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
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

print(str(trainingDataLabeled[0]))
trainingData = list(map(lambda z: z.data, trainingDataLabeled))
trainingLabels = list(map(lambda z: z.label, trainingDataLabeled))
testData= list(map(lambda z: z.data, testDataLabeled))
testLabels = list(map(lambda z: z.label, testDataLabeled))

print(str(trainingData[0]))
clf = NearestCentroid()
X = np.array(trainingData)
y = np.array(trainingLabels)
clf.fit(X, y)

correct = 0
wrong = 0
for index, image in enumerate(testData):
    prediction = clf.predict(image)
    if prediction == testLabels[index]:
        correct += 1
    else:
        wrong += 1

print("Correct: " + str(correct))
print("Wrong: " + str(wrong))

nc_classify(trainingData, testData, testLabels, trainingLabels)
