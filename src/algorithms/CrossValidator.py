from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def cross_val_score(validator, data, target, cv):
    for i in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)
        validator.fit(X_train, y_train)
        predictions = validator.predictAll(X_test)
        return accuracy_score(predictions, y_test, normalize=True)
