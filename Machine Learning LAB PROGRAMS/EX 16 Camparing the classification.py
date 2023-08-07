import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# Load the Iris dataset
iris = pd.read_csv("C:/Users/koppo/Downloads/IRIS.csv")
print(iris.head())

# Separate features and labels
x = iris.drop("species", axis=1)
y = iris["species"]

# Split the data into training and testing sets (test_size should be a value between 0 and 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize classifiers
decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

# Fit the classifiers to the training data
knearestclassifier.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
logisticregression.fit(x_train, y_train)
passiveAggressive.fit(x_train, y_train)

# Calculate the accuracy score for each classifier on the entire dataset
data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier",
                                       "Logistic Regression", "Passive Aggressive Classifier"],
         "Score": [knearestclassifier.score(x, y), decisiontree.score(x, y),
                   logisticregression.score(x, y), passiveAggressive.score(x, y)]}
score = pd.DataFrame(data1)
print(score)
