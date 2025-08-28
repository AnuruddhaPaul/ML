import numpy as np
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.activation = self._unit_step_function
    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y=np.where(y > 0, 1, 0)

        for _ in range(self.n_iter):
            for x, target in zip(X, y):
                update = self.eta * (target - self.predict(x))
                self.weights += update * x
                self.bias += update
        return self
    def net_input(self, X): 
        return np.dot(X, self.weights) + self.bias  
    def predict(self, X):        
        linear_output = self.net_input(X) 
        return self.activation(linear_output)    
    def predict_proba(self, X):
        linear_output = self.net_input(X)
        probabilities = 1 / (1 + np.exp(-linear_output))
        return probabilities
        
    
    