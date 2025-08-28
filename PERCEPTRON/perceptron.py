import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from perceptron_test import Perceptron
import matplotlib.pyplot as plt

# Generate a simple binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_clusters_per_class=1, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
# Initialize and train the Perceptron model
model = Perceptron(eta=0.01, n_iter=1000)
model.fit(X_train, y_train) 
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')



   

 

