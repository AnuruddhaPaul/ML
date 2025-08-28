import numpy as np
import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nb_test import NaiveBayes
from sklearn.naive_bayes import GaussianNB

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Naive Bayes
model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Custom Naive Bayes Accuracy:", accuracy(y_test, y_pred))

# Sklearn Naive Bayes
sklearn_model = GaussianNB()
sklearn_model.fit(X_train, y_train)
y_sklearn_pred = sklearn_model.predict(X_test)
print("Sklearn Naive Bayes Accuracy:", accuracy(y_test, y_sklearn_pred))

# Plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

plt.scatter(X[:, 2], X[:, 3], c=y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
# Compare predictions
print("Custom Model Predictions:", y_pred)
print("Sklearn Model Predictions:", y_sklearn_pred)
print("Predictions Match:", np.array_equal(y_pred, y_sklearn_pred))
print("Custom Model Probabilities:", model.predict_proba(X_test))