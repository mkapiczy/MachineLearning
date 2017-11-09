import sys
import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingImages)
        y = np.array(trainingLabels)
        neigh = KNeighborsClassifier(n_neighbors=2)
        neigh.fit(X, y)

        correct = 0
        wrong = 0
        for index, image in enumerate(testImages):
            print('\r{0:d}'.format(index), end='')
            sys.stdout.flush()
            prediction = neigh.predict(image)
            if prediction == testLabels[index]:
                correct += 1
            else:
                wrong += 1

        print("Correct: " + str(correct))
        print("Wrong: " + str(wrong))