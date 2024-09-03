import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    weights1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    weights2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return weights1, b1, weights2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(weights1, b1, weights2, b2, X):
    Z1 = weights1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = weights2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, weights1, weights2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dweights2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = weights2.T.dot(dZ2) * ReLU_deriv(Z1)
    dweights1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dweights1, db1, dweights2, db2

def update_params(weights1, b1, weights2, b2, dweights1, db1, dweights2, db2, alpha):
    weights1 = weights1 - alpha * dweights1
    b1 = b1 - alpha * db1    
    weights2 = weights2 - alpha * dweights2  
    b2 = b2 - alpha * db2    
    return weights1, b1, weights2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    weights1, b1, weights2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(weights1, b1, weights2, b2, X)
        dweights1, db1, dweights2, db2 = backward_prop(Z1, A1, Z2, A2, weights1, weights2, X, Y)
        weights1, b1, weights2, b2 = update_params(weights1, b1, weights2, b2, dweights1, db1, dweights2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return weights1, b1, weights2, b2

weights1, b1, weights2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, weights1, b1, weights2, b2):
    _, _, _, A2 = forward_prop(weights1, b1, weights2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, weights1, b1, weights2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], weights1, b1, weights2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, weights1, b1, weights2, b2)
test_prediction(1, weights1, b1, weights2, b2)
test_prediction(2, weights1, b1, weights2, b2)
test_prediction(3, weights1, b1, weights2, b2)

dev_predictions = make_predictions(X_dev, weights1, b1, weights2, b2)
get_accuracy(dev_predictions, Y_dev)