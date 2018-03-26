import numpy as np
import time

# Load Training DataSet
train_data = np.load('MNIST/train_data.npy')
train_labels = np.load('MNIST/train_labels.npy')

# Load Testing DataSet
test_data = np.load('MNIST/test_data.npy')
test_labels = np.load('MNIST/test_labels.npy')


# Finding Squared Euclidean Distance
def squared_dist(x, y):
    return np.sum(np.square(x - y))


# Find NN
def find_NN(x):
    # Compute distances from x to every row in train_data
    distances = [squared_dist(x, train_data[i,]) for i in range(len(train_labels))]
    # Get the index of the smallest distance
    return np.argmin(distances)


# NN Classifier
def NN_classifier(x):
    # Get the index of the the nearest neighbor
    index = find_NN(x)
    # Return its class
    return train_labels[index]


t_before = time.time()
test_predictions = [NN_classifier(test_data[i, ]) for i in range(len(test_labels))]
t_after = time.time()

# Compute the error
err_positions = np.not_equal(test_predictions, test_labels)
error = float(np.sum(err_positions))/len(test_labels)

# Error Rate
print("Error of nearest neighbor classifier: ", error)

# Time taken (Varies from system to system)
print("Classification time (seconds): ", t_after - t_before)

# Mine Output
# Error of nearest neighbor classifier:  0.046
# Classification time (seconds):  204.9625482559204
