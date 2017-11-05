import numpy as np
from scipy.io import loadmat

data = loadmat('./samples/ORL/orl_data.mat')['data']
labels = loadmat('./samples/ORL/orl_lbls.mat')['lbls']

print(data)
print(labels)