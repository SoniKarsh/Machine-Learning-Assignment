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


def NN_L1_squared_distance(x, y):
    return np.sum(np.absolute(x - y))


def NN_L1(trainX, trainY, testX):
    testYprediction = []
    for x in range(len(testX)):
        distances = [NN_L1_squared_distance(testX[x, ], trainX[i, ]) for i in range(len(trainX))]
        index = np.argmin(distances)
        testYprediction.append(trainY[index])

    return np.array(testYprediction)


def NN_L2_squared_distance(x, y):
    return np.sum(np.square(x - y))


def NN_L2(trainX, trainY, testX):
    testYprediction = []
    for x in range(len(testX)):
        distances = [NN_L2_squared_distance(testX[x, ], trainX[i, ]) for i in range(len(trainX))]
        index = np.argmin(distances)
        testYprediction.append(trainY[index])

    return np.array(testYprediction)


def error_rate(testy, testy_fit):
    return float(sum(testy != testy_fit))/len(testy)


def confusion(testy, testy_fit):
    confusionMatrix = np.zeros((3, 3))
    for i in range(len(testy)):
        if testy[i] != testy_fit[i]:
            print(int(testy[i]))
            confusionMatrix[int(testy[i])][int(testy_fit[i])] += 1
    return confusionMatrix


testy_L1 = NN_L1(trainX, trainY, testX)
testy_L2 = NN_L2(trainX, trainY, testX)

# Error Rate
print("Error rate of NN_L1: ", error_rate(testY, testy_L1))
print("Error rate of NN_L2: ", error_rate(testY, testy_L2))

L2_neo = confusion(testY, testy_L2)
L1_neo = confusion(testY, testy_L1)

# For Code Check
# assert( np.all(L1_neo == [[ 0.,  2.,  2.],[ 10.,  0.,  0.],[ 0.,  0.,  0.]]) )
# assert( np.all(L2_neo == [[ 0.,  1.,  2.],[ 10.,  0.,  0.],[ 0.,  0.,  0.]]) )
