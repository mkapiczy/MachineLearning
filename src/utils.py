import numpy as np


def chunks(l, n):
    chunks = []
    for i in range(0, len(l), n):
        chunks.append(l[i:i + n])
    return chunks


def to_zero_form(number, numberOfClasses):
    array = []
    for i in range(numberOfClasses):
        if i == number:
            array.append(1)
        else:
            array.append(-1)
    return array


def transformLabels(labels):
    labelsTranspose = labels.transpose()
    newLabels = []
    for label in labelsTranspose:
        newLabels.append(to_zero_form(label, len(np.unique(labels))))
    return newLabels


def addBias(vector):
    result = []
    for v in vector:
        result.append(v + [1])
    return result
