from loader import MNIST
import numpy as np

mndata = MNIST('./samples/MNIST/')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

print("Training images: " + str(len(trainingImages[1])))
print("Training labels: " + str(len(trainingLabels)))
X = np.array(trainingImages)
y = np.array(trainingLabels)

print(mndata.display(trainingImages[1]))
print(trainingLabels[1])