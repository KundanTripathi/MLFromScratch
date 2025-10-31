import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter= 1000):
        self.lr = lr
        self.n_iter = n_iter 
        self.w = None
        self.b = None 

    def fit(self, X_train, y_train):

        n_samples = X_train.shape[0]
        self.w = np.zeros(X_train.shape[1])
        self.b = 0 
        
        for _ in range(self.n_iter):
            predictions = np.dot(X_train, self.w) + self.b

            dw = (2/n_samples) * ( np.dot(X_train.T, (predictions - y_train)))
            db = (2/n_samples) * np.sum((predictions - y_train))

            self.w  = self.w - self.lr*dw 
            self.b = self.b - self.lr*db

    def predict(self,X):
        predictions = np.dot(X , self.w) + self.b
        return predictions
    
    def mse(self, predictions, y_true):
        mse = np.mean((predictions - y_true)**2)
        return mse 
    





    