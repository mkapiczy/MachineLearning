import warnings

import numpy as np
from matplotlib import pyplot, lines
from scipy.io import loadmat
from sklearn.decomposition import PCA

from algorithms.NearestSubclass.MyNearestSubclassCentroid import MyNearestSubclassCentroid
from orl.orl_preprocessor import preprocessData

indexcolors = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]

markers = [
    ".", ",", "o", "v", "^", "<", ">", "1", "2", "3",
    "4", "8", "s", "p", "P", "*", "h", "H", "+", "x",
    "X", "D", "d", "|", "_", lines.TICKLEFT, lines.TICKRIGHT, lines.TICKUP, lines.TICKDOWN, lines.CARETLEFT,
    lines.CARETRIGHT, lines.CARETUP, lines.CARETDOWN, lines.CARETLEFTBASE, lines.CARETRIGHTBASE, lines.CARETUPBASE,
    "$Y$", "$&$", "$S$", "$=$"
]

data = loadmat('../../samples/ORL/orl_data.mat')['data']
labels = loadmat('../../samples/ORL/orl_lbls.mat')['lbls']

trainingData, trainingLabels, testData, testLabels = preprocessData(data, labels)
X = trainingData
y = trainingLabels


def getColorMarkerFromLabel(label):
    return indexcolors[label], markers[label]


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    np.random.seed(0)

    X = np.array(X)
    y = np.array(y)

    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)

    X = np.array(pca_2d)
    clf = MyNearestSubclassCentroid(3)
    clf.fit(X, y)

    pca = PCA(n_components=2).fit(testData)
    pca_2d_test = pca.transform(testData)


    for i, x in enumerate(pca_2d_test):
        label = clf.predict(x)
        # label = y[i]
        color, marker = getColorMarkerFromLabel(label)

        pyplot.scatter(x[0], x[1], c=color, marker=marker)
    # for centroid in clf.centroids:
    #     centroidValue = np.array(centroid.value)
    #
    #     centroidLabel = centroid.label
    #     pyplot.scatter(centroidValue[0][0], centroidValue[0][1], 60, c='#e50000', marker='x')
    #     pyplot.annotate(str(centroidLabel), centroidValue[0], centroidValue[0])


    pyplot.show()
