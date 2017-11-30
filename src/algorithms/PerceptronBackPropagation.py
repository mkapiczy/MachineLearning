import warnings

from sklearn.neural_network import MLPClassifier
import numpy as np

from visual.confusion_matrix import createConfusionMatrix


# def validateNearestNeighbourHyperParameter(trainingData, trainingLabels, k):
#     print("Validate start")
#     clf = KNeighborsClassifier(k)
#     print("Clf created")
#     scores = cross_val_score(clf, trainingData, np.array(trainingLabels), cv=5)
#     print("cross val ended")
#     print(str(scores))

def test_perceptron_backpropagation(trainingImages, trainingLabels, testImages, testLabels, alpha=1e-5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.array(trainingImages)
        y = np.array(trainingLabels)

        clf = MLPClassifier(solver='lbfgs', alpha=alpha,
                            hidden_layer_sizes=(), random_state=1)
        clf.fit(X, y)

        correct = 0
        wrong = 0
        for index, image in enumerate(testImages):
            prediction = clf.predict(image)
            if prediction == testLabels[index]:
                correct += 1
            else:
                wrong += 1

        print("Correct: " + str(correct))
        print("Wrong: " + str(wrong))

        predictions = clf.predict(testImages)
        createConfusionMatrix(predictions, testLabels)

