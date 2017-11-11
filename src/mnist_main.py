from datetime import datetime

from algorithms.nc_classify import test_nc_classify, test_nc_classify_with_sklearn
from algorithms.nearest_neighbour_classify import test_neigh_classify_with_sklearn
from algorithms.nsc_classify import test_nsc_classify
from loader import MNIST

mndata = MNIST('../samples/MNIST/')

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

# pca = PCA(n_components=3)
# trainingImages = pca.fit_transform(trainingImages)
# testImages = pca.fit_transform(testImages)

print("Nearest centroid - my implementation")
startTime = datetime.now()
test_nc_classify(trainingImages, trainingLabels, testImages, testLabels)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

print("Nearest centroid - sklearn")
startTime = datetime.now()
test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

print("Nearest subclass centroid - my implementation - 2")
startTime = datetime.now()
test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 2)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

print("Nearest subclass centroid - my implementation - 3")
startTime = datetime.now()
test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 3)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))

print("Nearest subclass centroid - my implementation - 5")
startTime = datetime.now()
test_nsc_classify(trainingImages, trainingLabels, testImages, testLabels, 5)
timeElapsed = datetime.now() - startTime
print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))


# print("Nearest neighbours - sklearn")
# startTime = datetime.now()
# test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)
# timeElapsed = datetime.now() - startTime
# print('Execution time(hh:mm:ss.ms) {}'.format(timeElapsed))
