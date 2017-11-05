import numpy as np

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def devideIntoClasses(data, labels):
    numberOfClasses = len(sort_and_deduplicate(labels))
    dataIntoClasses = [[] for x in range(numberOfClasses)]
    for index, l in enumerate(labels):
        dataIntoClasses[l].append(data[index])
    dataIntoClasses = np.array(dataIntoClasses)
    return dataIntoClasses


def calculateCentroids(dataDividedInClasses, centroidFunction):
    centroids = [0 for x in range(len(dataDividedInClasses))]
    for index, singleClassData in enumerate(dataDividedInClasses):
        centroids[index] = centroidFunction(singleClassData)
    return centroids


def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(euclideanDistance(Number, valor)))
    return aux.index(min(aux))


def euclideanDistance(a, b):
    return np.linalg.norm(a - b)


def test_nc_classify(testData, testLabels, centroids, evaluationFunction):
    correct = 0
    wrong = 0

    for index, data in enumerate(testData):
        dataMeanValue = evaluationFunction(data)
        closestCentroidIndex = closest(centroids, dataMeanValue)
        if closestCentroidIndex == testLabels[index]:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))


def nc_classify(trainData, testData, testLabels, trainLabels, centroidFunction, evaluationFunction):
    dataInClasses = devideIntoClasses(trainData, trainLabels)
    centroids = calculateCentroids(dataInClasses, centroidFunction)
    print(str(centroids))
    test_nc_classify(testData, testLabels, centroids, evaluationFunction)
