import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different values of 'm' to explore
m_values = [1.1, 2.0, 3.0]

# Loop through each 'm' value and display the Fuzzy KNN decision boundaries
for m_value in m_values:
    # Generate fuzzy partition matrix using Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_train.T, 3, m=m_value, error=0.005, maxiter=1000)

    # Plot decision boundaries
    plt.figure(figsize=(8, 5))
    plt.title(f"Fuzzy KNN Decision Boundaries (m={m_value})")

    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', edgecolors='k')

    # Plot decision boundaries
    h = 0.02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict using the fuzzy partition matrix
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(np.c_[xx.ravel(), yy.ravel()].T, cntr, m=m_value, error=0.005, maxiter=1000)
    Z = np.argmax(u, axis=0)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
plt.show()
