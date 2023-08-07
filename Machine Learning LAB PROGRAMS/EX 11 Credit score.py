import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/koppo/Downloads/CREDITSCORE.csv")
print(data.head())

print(data.info())

# Convert "Credit_Mix" to numerical labels using label encoding
credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 3}
data["Credit_Mix"] = data["Credit_Mix"].map(credit_mix_mapping)

# Extract features and target variable
x = data[["Annual_Income", "Monthly_Inhand_Salary",
          "Num_Bank_Accounts", "Num_Credit_Card",
          "Interest_Rate", "Num_of_Loan",
          "Delay_from_due_date", "Num_of_Delayed_Payment",
          "Credit_Mix", "Outstanding_Debt",
          "Credit_History_Age", "Monthly_Balance"]].values
y = data["Credit_Score"].values

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

# Create and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Make predictions on new data
print("Credit Score Prediction:")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad, Standard, Good): ")
i = credit_mix_mapping.get(i, None)
if i is None:
    print("Invalid Credit Mix input. Please enter Bad, Standard, or Good.")
    exit()

j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
predicted_credit_score = model.predict(features)
print("Predicted Credit Score =", predicted_credit_score[0])

# Evaluate the model on the test set
y_pred = model.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
print("Model Accuracy on Test Set:", accuracy)

classification_rep = classification_report(ytest, y_pred)
print("Classification Report:\n", classification_rep)
