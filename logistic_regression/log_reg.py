import numpy as np

class LogisticRegression:
    """
    Logistic regression classifier
    Assume all inputs are tensors or numpy arrays.
    """
    def __init__(self, batch_size = 2, lr = 0.01, thres = 0.5, n_iter = 100):
        self.batch_size = batch_size
        self.lr = lr 
        self.thres = thres
        self.n_iter = n_iter
        self.w = None 
        self.b = None 

    def dW(self, X, y_pred, y):
        """
        Compute the gradient of the loss with respect to weights.""
        """
        m = X.shape[0]
        return (1 / m) * np.dot(X.T, (y_pred - y))

    def dB(self, y_pred, y):
        """
        Compute the gradient of the loss with respect to bias.
        """
        m = len(y)
        return (1 / m) * np.sum(y_pred - y)
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        """
        
        return (1/(1 + np.exp(-z)))

    def predict_proba(self, X):
        """
        Compute the predicted probabilities.
        """
        #self.w = np.zeros(X.shape[1])
        #b = 0
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        """
        Classify the input data, i.e. predict the binary labels.
        """
        return (self.predict_proba(X) >= self.thres).astype(int)
        
       

    def compute_loss(self, y, y_pred):
        """
        Compute the binary cross-entropy loss.
       -1/m*(ylog(y') + (1-y)log(1-y'))
        actual = 0 , prediction =1 
        
        """
        
        return -1/len(y)*(np.dot(y.T, np.log(1-y_pred)) + np.dot((1- y).T, np.log(1-y_pred)))

    def train_sgd(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model using stochastic gradient descent.
        """
        self.w = np.zeros(X_train.shape[1])
        self.b = 1
        for _ in range(self.n_iter):
            #ind = np.random.choice(X_train.shape[0])
            indices = np.random.permutation(X_train.shape[0])
            
            for idx in indices:
                # Select single sample
                xi = X_train[idx:idx+1]
                yi = y_train[idx:idx+1]

                y_pred = self.predict_proba(xi)
                dw = self.dW(xi, y_pred, yi)
                db = self.dB(y_pred, yi)
                
                self.w = self.w - self.lr* dw
                self.b = self.b - self.lr*db
                print(self.compute_loss(y_train, self.predict_proba(X_train)))
        y_pred = self.predict_proba(X_train)
        fin_pred =  self.predict(X_train)      
        print(self.w)
        print(self.b) 
        print(y_pred)
        print(fin_pred)

    def train_grad(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model using stochastic gradient descent.
        """
        self.w = np.zeros(X_train.shape[1])
        self.b = 1
        for _ in range(self.n_iter):
            y_pred = self.predict_proba(X_train)
            dw = self.dW(X_train, y_pred, y_train)
            db = self.dB(y_pred, y_train)
            
            self.w = self.w - self.lr* dw
            self.b = self.b - self.lr*db
            print(self.compute_loss(y_train, y_pred))
        y_pred = self.predict_proba(X_train)
        fin_pred =  self.predict(X_train)      
        print(self.w)
        print(self.b) 
        print(y_pred)
        print(fin_pred)

    def train_batch(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model using stochastic gradient descent.
        """
        self.w = np.zeros(X_train.shape[1])
        self.b = 1
        for _ in range(self.n_iter):
            '''
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]
            '''
            for idx in range(0, X_train.shape[0], self.batch_size):
                start =idx
                end = idx + self.batch_size
                # Select single sample
                xi = X_train[start:end]
                yi = y_train[start:end]

                y_pred = self.predict_proba(xi)
                dw = self.dW(xi, y_pred, yi)
                db = self.dB(y_pred, yi)
                
                self.w = self.w - self.lr* dw
                self.b = self.b - self.lr*db
                print(self.compute_loss(y_train, self.predict_proba(X_train)))
        y_pred = self.predict_proba(X_train)
        fin_pred =  self.predict(X_train)      
        print(self.w)
        print(self.b) 
        print(y_pred)
        print(fin_pred)
                
X = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [2.0, 2.0], [2.5, 1.5]])
y = np.array([0, 0, 0, 1, 1])
LR = LogisticRegression(batch_size = 1, lr = 0.001, thres = 0.5, n_iter = 50000)
LR.train_batch(X_train=X, y_train=y, X_val=None, y_val=None)