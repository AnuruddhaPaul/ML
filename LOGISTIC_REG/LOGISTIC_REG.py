import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LOGISTIC_REG_TEST import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0.0
    precision = true_positives / predicted_positives
    return precision

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0.0
    recall = true_positives / actual_positives
    return recall

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if (prec + rec) == 0:
        return 0.0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1234)
model = LogisticRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Custom Logistic Regression")
print("Accuracy: ", accuracy_score(y_test, predictions))    
print("Precision: ", precision_score(y_test, predictions))
print("Recall: ", recall_score(y_test, predictions))    
print("F1 Score: ", f1_score(y_test, predictions))    
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Weights: ", model.weights)
print("Bias: ", model.bias)
plot_decision_boundary(X, y, model)
# sklearn       
model_sk = SKLogisticRegression()
model_sk.fit(X_train, y_train)
predictions_sk = model_sk.predict(X_test)
print("\nSKLearn Logistic Regression")
print("Accuracy: ", accuracy_score(y_test, predictions_sk))
print("Precision: ", precision_score(y_test, predictions_sk))
print("Recall: ", recall_score(y_test, predictions_sk))
print("F1 Score: ", f1_score(y_test, predictions_sk))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions_sk))
print("Weights: ", model_sk.coef_)
print("Bias: ", model_sk.intercept_)
plot_decision_boundary(X, y, model_sk)

