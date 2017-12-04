from datetime import datetime

from algorithms.PerceptronBP.perceptron_bp_test import test_perceptron_bp
from algorithms.PerceptronBackPropagation import test_perceptron_backpropagation
from algorithms.PerceptronMSE.PerceptronMSEClassifier import PerceptronMSEClassifier
from algorithms.PerceptronMSE.perceptron_mse_test import test_perceptron_mse, validateHyperParameter
from algorithms.nc_classify import test_nc_classify, test_nc_classify_with_sklearn
from algorithms.nearest_neighbour_classify import test_neigh_classify_with_sklearn, \
    validateNearestNeighbourHyperParameter
from algorithms.nsc_classify import test_nsc_classify
from loader import MNIST

mndata = MNIST('../samples/MNIST/')

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()
#
# print("Nearest centroid - my implementation")
# startTime = datetime.now()
# test_nc_classify(trainingImages, trainingLabels, testImages, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# print("Nearest centroid - sklearn")
# startTime = datetime.now()
# test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#

# print("Nearest subclass centroid - my implementation - 1")
# validateHyperParameter(trainingImages, trainingLabels, 1)
# startTime = datetime.now()
# # test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 1)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
# 0.809083333333
# print("Nearest subclass centroid - my implementation - 2")
# validateHyperParameter(trainingImages, trainingLabels, 2)
# startTime = datetime.now()
# # test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest subclass centroid - my implementation - 3")
# validateHyperParameter(trainingImages, trainingLabels, 3)
# startTime = datetime.now()
# # test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 3)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
# #
# print("Nearest subclass centroid - my implementation - 5")
# validateHyperParameter(trainingImages, trainingLabels, 5)
# startTime = datetime.now()
# # test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 5)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


# print("Nearest neighbours - sklearn k-1")
# startTime = datetime.now()
# # validateNearestNeighbourHyperParameter(trainingImages, trainingLabels, 1)
# test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels, 1)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# print("Nearest neighbours - sklearn k-2")
# startTime = datetime.now()
# validateNearestNeighbourHyperParameter(trainingImages, trainingLabels, 2)
# test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# print("Nearest neighbours - sklearn k-3")
# startTime = datetime.now()
# validateNearestNeighbourHyperParameter(trainingImages, trainingLabels, 3)
# # test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# print("Backpropagation")
# startTime = datetime.now()
# # validateNearestNeighbourHyperParameter(trainingImages, trainingLabels, 3)
# test_perceptron_backpropagation(trainingImages, trainingLabels, testImages, testLabels, 0.1)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

# ------- MSE Perceptron --------
# validateHyperParameter(trainingImages, trainingLabels)
test_perceptron_bp(trainingImages, trainingLabels, testImages, testLabels)
# test_perceptron_mse(trainingImages, trainingLabels, testImages, testLabels)