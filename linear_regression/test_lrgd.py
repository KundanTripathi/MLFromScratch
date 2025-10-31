import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_regression
from lrgd import LinearRegressionGD
import matplotlib.pyplot as plt 


X , y  = make_regression(n_samples = 100, n_features = 1, n_targets = 1, noise =20, random_state = 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

LR = LinearRegressionGD()

LR.fit(X_train, y_train)
pred = LR.predict(X_test)
acc = LR.mse(pred, y_test)

plt.figure(figsize=(10,8))
plt.scatter(X_test, y_test,color='b', marker= 'o', s= 30, label='Data points')
plt.plot(X_test, pred, color='red', label='Linear regression line')
plt.show()


print(pred)
print(acc)
