import numpy as np
class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.variance = {}
        self.epsilon = 1e-6  # To prevent division by zero
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.class_priors[cls] = X_c.shape[0] / n_samples
            self.mean[cls] = np.mean(X_c, axis=0)
            self.variance[cls] = np.var(X_c, axis=0) + self.epsilon
    def _calculate_likelihood(self, cls, x):
        mean = self.mean[cls]
        variance = self.variance[cls]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        likelihood = numerator / denominator
        return likelihood
    def _calculate_posterior(self, x):
        posteriors = {}
        for cls in self.classes:
            likelihood = self._calculate_likelihood(cls, x)
            prior = self.class_priors[cls]
            posterior = likelihood * prior
            posteriors[cls] = posterior
        return posteriors
    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            predicted_class = self.classes[np.argmax(list(posteriors.values())) % len(self.classes)]
            y_pred.append(predicted_class)
        return np.array(y_pred)
    def predict_proba(self, X):
        proba = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            total = sum(posteriors.values())
            probs = {cls: posteriors[cls] / total for cls in self.classes}
            proba.append(probs)
        return proba
