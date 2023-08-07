import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset using pandas
iris = pd.read_csv("C:/Users/koppo/Downloads/IRIS.csv")

print(iris.head())
print()
print(iris.describe())
print("Target Labels", iris["species"].unique())

import plotly.io as io
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Split the data into features (x) and labels (y)
x = iris.drop("species", axis=1)
y = iris["species"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and fit the KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Make a prediction for a new data point
x_new = np.array([[6, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))

