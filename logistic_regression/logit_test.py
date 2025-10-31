import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from logit import LogisticRegression

X, y = make_classification(n_classes=2, n_samples = 500, n_features=2, n_informative=2, n_redundant=0, flip_y=0.01, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=123)

LR = LogisticRegression()

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)

print("prediction:" , y_pred)
print("actual    :" , y_test)