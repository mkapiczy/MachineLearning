from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from random import randint


def validateHyperParameter(trainingData, trainingLabels, clf):
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    scores = cross_val_score(clf, trainingData, trainingLabels, cv=5)
    print(str(scores))

def cross_val_score(validator, data, target, cv):
    scores = []
    for i in range(cv):
        randomInt = randint(0, 99)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=randomInt, stratify=target)
        validator.fit(X_train, y_train)
        predictions = validator.predict(X_test)
        scores.append(accuracy_score(predictions, y_test, normalize=True))
    return np.mean(scores)
