import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Load the training and test datasets
train_data = pd.read_csv('./MNIST/train.csv')
test_data = pd.read_csv('./MNIST/test.csv')

X_train = train_data.drop('label', axis=1).values 
y_train = train_data['label'].values  

X_test = test_data.values

X_train = X_train / 255.0
X_test = X_test / 255.0

perceptron = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=1000, solver='sgd', random_state=42)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

def test_prediction(index):
    current_image = X_test[index].reshape(28, 28)
    prediction = perceptron.predict(X_test[index].reshape(1, -1))
    print(f"Prediction: {prediction[0]}")

    # Display the image
    plt.imshow(current_image, cmap='gray')
    plt.title(f'Prediction: {prediction[0]}')
    plt.axis('off')
    plt.show()

test_prediction(0)
