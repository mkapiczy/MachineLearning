import warnings

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from algorithms.CrossValidator import cross_val_score
from visual.confusion_matrix import createConfusionMatrix


def validateNearestNeighbourHyperParameter(trainingData, trainingLabels, k):
    print("Validate start")
    clf = KNeighborsClassifier(k)
    print("Clf created")
    scores = cross_val_score(clf, trainingData, np.array(trainingLabels), cv=5)
    print("cross val ended")
    print(str(scores))

def test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels, K):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingImages)
        y = np.array(trainingLabels)

        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(X, y)

        # correct = 0
        # wrong = 0
        # for index, image in enumerate(testImages):
        #     print('\r{0:d}'.format(index), end='')
        #     sys.stdout.flush()
        #     prediction = neigh.predict(image)
        #     if prediction == testLabels[index]:
        #         correct += 1
        #     else:
        #         wrong += 1
        #
        # print("Correct: " + str(correct))
        # print("Wrong: " + str(wrong))

        predictions = neigh.predict(testImages)
        createConfusionMatrix(predictions, testLabels)