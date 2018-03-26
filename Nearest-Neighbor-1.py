import numpy as np
import matplotlib.pyplot as plt

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


# Define a function that displays a digit given its vector representation
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap="gray")
    plt.show()
    return


# Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
def vis_image(index, dataset="train"):
    if dataset== "train":
        show_digit(train_data[index,])
        label = train_labels[index]
    else:
        show_digit(test_data[index,])
        label = test_labels[index]
    print("Label " + str(label))
    return


# Check for success and failure case
def checkForSuccessFailureCase(train_label, test_label, test_index):
    if train_label == test_label:
        print("A success case:")
        print("NN classification: ", NN_classifier(test_data[test_index,]))
        print("True label: ", test_labels[test_index])
        print("The test image:")
        vis_image(test_index, "test")
        print("The corresponding nearest neighbor image:")
        vis_image(find_NN(test_data[test_index,]), "train")
    elif train_label != test_label:
        print("A failure case:")
        print("NN classification: ", NN_classifier(test_data[test_index,]))
        print("True label: ", test_labels[test_index])
        print("The test image:")
        vis_image(test_index, "test")
        print("The corresponding nearest neighbor image:")
        vis_image(find_NN(test_data[test_index,]), "train")
    else:
        print("Error Occurred!!!")

# User input for test point
test_index = int(input("Enter the test Point \n"))

# Getting test label of the test index
test_label = test_labels[test_index]

# Getting train label which NN will guess for our test point by calling below method
train_label = NN_classifier(test_data[100])

checkForSuccessFailureCase(train_label, test_label, test_index)
