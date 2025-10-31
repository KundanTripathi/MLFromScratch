from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from knn import KNN
import numpy as np

def test_knn():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create an instance of KNN with k=3
    model = KNN(k=3)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Check if predictions match the expected output
    assert len(predictions) == len(y_test), "Number of predictions does not match number of test samples"
    #assert all(isinstance(pred, int) for pred in predictions), "All predictions should be integers (class labels)"

    
    accuracy = np.sum(predictions == y_test) / len(y_test)
    
    # Print results
    print("Predictions:", predictions)
    print("True labels:", y_test)
    print("Accuracy:", accuracy)

   

if __name__ == "__main__":
    test_knn()
    