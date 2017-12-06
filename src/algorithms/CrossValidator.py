import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def validateHyperParameter(trainingData, trainingLabels, clf):
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    scores = cross_val_score(clf, trainingData, trainingLabels, cv=5)
    print(str(scores))

def cross_val_score(validator, data, target, cv):
    scores = []
    kf = KFold(n_splits=cv)
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        validator.fit(X_train, y_train)
        predictions = validator.predict(X_test)
        scores.append(accuracy_score(predictions, y_test, normalize=True))
    return np.mean(scores)
