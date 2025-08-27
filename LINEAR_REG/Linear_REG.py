import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_Reg_test import LinearRegression
from sklearn.linear_model import LinearRegression as SKLinearRegression
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean)**2)
    residual_variance = np.sum((y_true - y_pred)**2)
    return 1 - (residual_variance / total_variance)
X,y=datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)
regressor=LinearRegression(lr=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predicted=regressor.predict(X_test)
print(predicted)
print(regressor.weights, regressor.bias)
print("MSE: ", mean_squared_error(y_test, predicted)
      , "R2 Score: ", r2_score(y_test, predicted))
#sklearn
regressor_sk=SKLinearRegression()
regressor_sk.fit(X_train, y_train)
print(regressor_sk.coef_, regressor_sk.intercept_)
predicted_sk=regressor_sk.predict(X_test)
print(predicted_sk)
print("MSE: ", mean_squared_error(y_test, predicted_sk)
      , "R2 Score: ", r2_score(y_test, predicted_sk))
#plot
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, predicted, color='blue', label='Custom Model')
plt.plot(X_test, predicted_sk, color='green', linestyle='dashed', label='SKLearn Model')
plt.legend()
plt.show()
