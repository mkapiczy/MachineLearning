from datetime import datetime

from scipy.io import loadmat

from algorithms.nc_classify import test_nc_classify, test_nc_classify_with_sklearn
from algorithms.nsc_classify import validateHyperParameter, test_nsc_classify
from orl.orl_preprocessor import preprocessData

data = loadmat('../samples/ORL/orl_data.mat')['data']
labels = loadmat('../samples/ORL/orl_lbls.mat')['lbls']

trainingData, trainingLabels, testData, testLabels = preprocessData(data, labels)
data = trainingData + testData
labels = trainingLabels + testLabels



#
# print("Nearest centroid - my implementation")
# startTime = datetime.now()
# test_nc_classify(trainingData, trainingLabels, testData, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
# print("Nearest centroid - sklearn")
# startTime = datetime.now()
# test_nc_classify_with_sklearn(trainingData, trainingLabels, testData, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


print("Nearest subclass centroid - my implementation - 1")
validateHyperParameter(data, labels, 1)
startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 1)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

print("Nearest subclass centroid - my implementation - 2")
validateHyperParameter(data, labels, 2)
startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 2)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
#
print("Nearest subclass centroid - my implementation - 3")
validateHyperParameter(data, labels, 3)
startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 3)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
# #
print("Nearest subclass centroid - my implementation - 5")
validateHyperParameter(data, labels, 5)
startTime = datetime.now()
# test_nsc_classify(trainingData, trainingLabels, testData, testLabels, 5)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


# print("Nearest neighbours - sklearn")
# startTime = datetime.now()
# test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels, 2)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))