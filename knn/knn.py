
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k 

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 

    def predict(self, X):
        # Placeholder for prediction logic
        # Calculate distance of each point in X to all points in X_train
        # Sort distances and get the indicies of the k nearest neigbour 
        # Return the most common class using max vote 
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
            indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in indices]

            counter = {}
            for label in nearest_labels:
                if label in counter:
                    counter[label] += 1
                else:
                    counter[label] = 1
            print("Distances:", distances)
            print("Indices of Nearest Neighbors:", indices)
            print("Nearest Labels:", nearest_labels)

            nearest_label = max(counter, key=counter.get)
            predictions.append(nearest_label)
        return predictions