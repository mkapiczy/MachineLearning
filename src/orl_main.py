from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

from algorithms.CrossValidator import validateHyperParameter
from algorithms.NearestCentroid.MyNearestCentroid import MyNearestCentroid
from algorithms.NearestCentroid.nc_classify import test_nc_classify, test_nc_classify_with_sklearn
from algorithms.NearestNeighbours.nearest_neighbour_classify import test_neigh_classify
from algorithms.NearestSubclass.MyNearestSubclassCentroid import MyNearestSubclassCentroid
from algorithms.NearestSubclass.nsc_classify import test_nsc_classify
from algorithms.PerceptronBP.perceptron_bp_test import test_perceptron_bp
from algorithms.PerceptronMSE.PerceptronMSEClassifier import PerceptronMSEClassifier
from algorithms.PerceptronMSE.perceptron_mse_test import test_perceptron_mse
from loader import MNIST
import numpy as np
from scipy.io import loadmat


from orl.orl_preprocessor import preprocessData

from algorithms.PerceptronBP.PerceptronBPClassifier import PerceptronBPClassifier

data = loadmat('../samples/ORL/orl_data.mat')['data']
labels = loadmat('../samples/ORL/orl_lbls.mat')['lbls']

trainingData, trainingLabels, testData, testLabels = preprocessData(data, labels)
data = trainingData + testData
labels = trainingLabels + testLabels


# # ------- PCA ---------
# pca = PCA(n_components=2).fit(np.array(trainingData))
# trainingData = pca.transform(np.array(trainingData))
#
# pca = PCA(n_components=2).fit(np.array(testData))
# testData = pca.transform(np.array(testData))
#
# pca = PCA(n_components=2).fit(np.array(data))
# data = pca.transform(np.array(data))
# # ------- PCA ---------



# # ------- NEAREST CENTROID MY IMPLEMENTATION --------
# print("Nearest centroid - my implementation")
# validateHyperParameter(data, labels, MyNearestCentroid())
# startTime = datetime.now()
# test_nc_classify(trainingData, trainingLabels, testData, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# # ------- NEAREST CENTROID SKLEARN --------
# print("Nearest centroid - sklearn")
# validateHyperParameter(data, labels, NearestCentroid())
# startTime = datetime.now()
# test_nc_classify_with_sklearn(trainingData, trainingLabels, testData, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))



# ------- NEAREST SUBCLASS CLASSIFIERS --------

# print("Nearest subclass centroid - my implementation - 1")
# validateHyperParameter(data, labels, MyNearestSubclassCentroid(1))
# startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 1)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# print("Nearest subclass centroid - my implementation - 2")
# validateHyperParameter(data, labels, MyNearestSubclassCentroid(2))
# startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest subclass centroid - my implementation - 3")
# validateHyperParameter(data, labels, MyNearestSubclassCentroid(3))
# startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 3)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest subclass centroid - my implementation - 5")
# validateHyperParameter(data, labels, MyNearestSubclassCentroid(5))
# startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 5)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# ------- K NEAREST NEIGHBOURS CLASSIFIERS --------

# print("Nearest neighbours - sklearn k-1")
# validateHyperParameter(data, labels, KNeighborsClassifier(n_neighbors=1))
# startTime = datetime.now()
# test_neigh_classify(trainingData, trainingLabels, testData, testLabels, 1)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest neighbours - sklearn k-2")
# validateHyperParameter(data, labels, KNeighborsClassifier(n_neighbors=2))
# startTime = datetime.now()
# test_neigh_classify(trainingData, trainingLabels, testData, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest neighbours - sklearn k-3")
# validateHyperParameter(data, labels, KNeighborsClassifier(n_neighbors=3))
# startTime = datetime.now()
# test_neigh_classify(trainingData, trainingLabels, testData, testLabels, 3)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest neighbours - sklearn k-5")
# validateHyperParameter(data, labels, KNeighborsClassifier(n_neighbors=5))
# startTime = datetime.now()
# test_neigh_classify(trainingData, trainingLabels, testData, testLabels, 5)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


# ------- Perceptron Backpropagation --------
print("Perceptron Backpropagation")
validateHyperParameter(data, labels, PerceptronBPClassifier())
startTime = datetime.now()
test_perceptron_bp(trainingData, trainingLabels, testData, testLabels)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


# # ------- MSE Perceptron --------
# print("Perceptron MSE")
# validateHyperParameter(data, labels, PerceptronMSEClassifier())
# startTime = datetime.now()
# test_perceptron_mse(trainingData, trainingLabels, testData, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))