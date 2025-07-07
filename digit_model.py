# KNN is a simple machine learning algorithm used for classification and regression
# Compares the new 28 x 28 image of a digit to a big dataset, like MNIST, of labeled digit images
# Finds K most similar images
# Returns the majority label
# I decided to switch from Tesseract to KNN because Tesseract is designed for text

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Loads MNIST dataset
# 70,000 handwritten digits
print("Loading MNIST...")
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True, as_frame = False)

# Normalizes pixel values tp [0, 1]
X = X / 255.0

# Uses only didgits 1-9
mask = y != '0'
X = X[mask]
y = y[mask]

# Trains KNN classifier
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)

# Saves model to disk
joblib.dump(knn, "knn_digit_model.pkl")
print("Model saved as knn_digit_model.pkl")