from loader import MNIST
from nc_classify import nc_classify

mndata = MNIST('./samples/MNIST/')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()


def meanValue(data):
    dataSum = 0
    for elem in data:
        dataSum += elem
    return dataSum / len(data)


def classMeanValue(classData):
    dataSum = 0
    for singleImageData in classData:
        dataSum += meanValue(singleImageData)
    return dataSum / len(classData)


nc_classify(trainingImages, testImages, testLabels, trainingLabels, classMeanValue, meanValue)
