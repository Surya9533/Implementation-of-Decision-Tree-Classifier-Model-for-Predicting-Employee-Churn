# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SURYA R
RegisterNumber:212224040339
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])


x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]


y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)


dt = DecisionTreeClassifier(criterion="entropy")


dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


sample = pd.DataFrame([[0.5, 0.8, 9, 260, 6, 0, 1, 2]],
                      columns=x.columns)

print("Prediction:", dt.predict(sample))

plt.figure(figsize=(12, 8))
plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["Stayed", "Left"],
    filled=True
)

plt.show()
```
## Output:
<img width="598" height="877" alt="image" src="https://github.com/user-attachments/assets/bd89046f-39b3-4937-b8df-9423032126f5" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
