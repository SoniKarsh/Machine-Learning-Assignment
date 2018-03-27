import numpy as np
import time

# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']

data = np.loadtxt('column_3C.dat', converters={6: lambda s: labels.index(s)})
x = data[:, 0:6]
y = data[:, 6]

# Divide into training and test set
training_indices = list(range(0, 20)) + list(range(40, 188)) + list(range(230, 310))
test_indices = list(range(20, 40)) + list(range(188, 230))

trainX = x[training_indices, :]
trainY = y[training_indices]
testX = x[test_indices, :]
testY = y[test_indices]


def NN_L2_squared_distance(x, y):
    return np.sum(np.square(x - y))


def NN_L2(trainX, trainY, testX):
    testYprediction = []
    for x in range(len(testX)):
        distances = [NN_L2_squared_distance(testX[x, ], trainX[i, ]) for i in range(len(trainX))]
        index = np.argmin(distances)
        testYprediction.append(trainY[index])

    return np.array(testYprediction)

testy_L2 = NN_L2(trainX, trainY, testX)
print(type(testy_L2))
print(len(testy_L2))
print(testy_L2[50:60])

# For checking your code
# assert(type(testy_L2).__name__ == 'ndarray')
# assert(len(testy_L2) == 62)
# assert(np.all(testy_L2[50:60] == [0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.]))
# assert(np.all(testy_L2[0:10] == [0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.]))

