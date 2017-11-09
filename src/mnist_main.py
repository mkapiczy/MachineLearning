from algorithms.nc_classify import nc_classify, test_nc_classify_with_sklearn
from algorithms.nearest_neighbour_classify import test_neigh_classify_with_sklearn
from loader import MNIST

mndata = MNIST('../samples/MNIST/')

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

# pca = PCA(n_components=3)
# trainingImages = pca.fit_transform(trainingImages)
# testImages = pca.fit_transform(testImages)

print("Nearest centroid - my implementation")
nc_classify(trainingImages, testImages, testLabels, trainingLabels)
print("Nearest centroid - sklearn")
test_nc_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)
print("Nearest neighbours - sklearn")
test_neigh_classify_with_sklearn(trainingImages, trainingLabels, testImages, testLabels)
