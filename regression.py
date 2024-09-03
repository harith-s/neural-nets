import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('/MNIST/train.csv')

# Split data into features and labels
X = data.drop('label', axis=1).values  # Drop the label column, rest are features
y = data['label'].values  # Labels (digit class)

# Normalize the data (pixel values are between 0 and 255)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")


def test_prediction(index):
    current_image = X_test[index].reshape(28, 28)
    prediction = model.predict(X_test[index].reshape(1, -1))
    label = y_test[index]
    print(f"Prediction: {prediction[0]}")
    print(f"Label: {label}")

    # Display the image
    plt.imshow(current_image, cmap='gray')
    plt.title(f'Prediction: {prediction[0]}, Label: {label}')
    plt.axis('off')
    plt.show()

# Test the prediction for an example index
test_prediction(0)
