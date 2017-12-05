import warnings

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from visual.confusion_matrix import createConfusionMatrix, printCorrectWrong


def test_neigh_classify(trainingImages, trainingLabels, testImages, testLabels, K):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingImages)
        y = np.array(trainingLabels)

        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(X, y)

        predictions = neigh.predict(testImages)
        printCorrectWrong(predictions, testLabels)
        # createConfusionMatrix(predictions, testLabels)