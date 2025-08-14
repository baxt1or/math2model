import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

class LogisticRegression:
    def __init__(self, lr, epochs):
        self.lr = lr 
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def fit(self, X, y):
        n_samples, n_feature = X.shape 
        self.weights = np.zeros(n_feature)
        self.bias = 0

        for _ in range(self.epochs):
            
            z = self._predict(X)
            pred = self.sigmoid(z)
            
            residuals = pred - y
            dw = (1/n_samples) *np.dot(X.T, residuals)
            db = (1/n_samples) * np.sum(residuals)

            self.weights -= self.lr * dw 
            self.bias -= self.lr * db 
    
    def predict_proba(self, X):
        return self.sigmoid(self._predict(X))
    
    def predict(self, X):
        return (self.sigmoid(self._predict(X)) >= 0.5).astype(int)
    


if __name__ == '__main__':

    data = load_iris()
    iris_data = data.get("data")
    target = data.get("target")

    model = LogisticRegression(lr=0.01, epochs=1000)
    model.fit(iris_data, target)
    pred = model.predict(iris_data)

    print("Accuracy:", accuracy_score(target, pred))