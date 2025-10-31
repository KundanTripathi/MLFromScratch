import numpy as np 

class LogisticRegression:
    def __init__(self, lr=0.001, n_iter = 1000):
        self.lr = lr 
        self.n_iter = n_iter
        self.w = None
        self.b =  None 

    def _sigmoid(self, z):
        return 1/ (1 + np.exp(-z))
    
    def _linear(self, X_train, w , b):
        return np.dot(X_train, w) + b
    
    def _initialize_parameters(self, X_train):
        w =  np.zeros(X_train.shape[1])
        b = 0
        return w , b

    def gradient(self, X_train, y_train, y_pred):
        dw = (1/X_train.shape[0]) * np.dot(X_train.T,(y_pred - y_train))
        db = (1/X_train.shape[0]) * np.sum(y_pred - y_train)
        return dw, db 
    
    def _update_parameters(self, w, dw, b, db, lr):
        w = w - lr*dw
        b = b - lr*db
        return w, b
    
    def fit(self, X_train, y_train):
        ''' " initialize parameter"
        " create training loop"
        " generate prediction using sigmoid and linear"
        " calculate gradient"
        " update w & b"
        '''
        self.w, self.b =  self._initialize_parameters(X_train)

        for _ in range(self.n_iter):

            y_pred = self._sigmoid(self._linear(X_train, self.w, self.b))
            dw , db = self.gradient(X_train, y_train, y_pred)
            self.w , self.b = self._update_parameters(self.w, dw, self.b, db, self.lr)

    def predict(self, X, threshold=0.5):
        y_pred = self._sigmoid(self._linear(X, self.w, self.b))
        return (y_pred>= threshold).astype(int)
    



    





    
