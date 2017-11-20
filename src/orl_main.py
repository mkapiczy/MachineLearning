from scipy.io import loadmat

from orl.orl_preprocessor import preprocessData

data = loadmat('../samples/ORL/orl_data.mat')['data']
labels = loadmat('../samples/ORL/orl_lbls.mat')['lbls']

trainingData, trainingLabels, testData, testLabels = preprocessData(data, labels)

# print("Nearest centroid - my implementation")
# nc_classify(trainingData, testData, testLabels, trainingLabels)
# print("Nearest centroid - sklearn")
# test_nc_classify_with_sklearn(trainingData, trainingLabels, testData, testLabels)
