import numpy as np


class LinearRegression: 
    def __init__(self, lr, epochs):
        self.lr = lr 
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def predict(self, X):
        return np.dot(X,self.weights) + self.bias
    
    def fit(self, X, y):
        m, n = X.shape 
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            
            pred = self.predict(X)
            residuals = pred - y

            dw = (1/m) * np.dot(X.T, residuals)
            db = (1/m) * np.sum(residuals)

            self.weights -= self.lr * dw 
            self.bias -= self.lr * db


if __name__ == '__main__':

    model = LinearRegression()

    X = np.array([ 
        []
    ])
    y = np.array([])



