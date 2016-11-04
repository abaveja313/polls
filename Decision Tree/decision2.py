from __future__ import print_function

__author__ = "Amrit Baveja"

import numpy as np
import pandas as pd
from sklearn import tree

<<<<<<< HEAD
train_df = pd.read_csv('model_new_new.csv')
# print(train_df)
y = targets = labels = train_df['Outcome'].values
y = np.array(y)
columns = ["Polls", "Results of Last Election"]
=======
train_df = pd.read_csv('florida/florida2.csv')
# print(train_df)
y = targets = labels = train_df['Results of Last election'].values
y = np.array(y)
columns = ["Average of last 3 elections", "Average of last 5 elections",
           "Average of polls 1 mo before election (>0 = Repub)", "% of registered republicans",
           "% of registered democrats", "State unemployment rate", "Party of governers"]
>>>>>>> 6b7f67e5c85747354cf8c8924176d441a10ccfcc

features = train_df[list(columns)].values
features = np.array(features)
print("Y data \n" + str(y))
print("------------------------")
print("X data \n" + str(features))
print("-------------------------")
X = features
<<<<<<< HEAD
clf = tree.DecisionTreeClassifier(criterion="entropy")
=======
clf = tree.DecisionTreeClassifier(criterion="entropy", max_features=3)
>>>>>>> 6b7f67e5c85747354cf8c8924176d441a10ccfcc
clf = clf.fit(X, y)
print("X shape: " + str(X.shape))
print("Y shape: " + str(y.shape))
print("--------------------------")
f = tree.export_graphviz(clf, out_file="decisiontree.dot", feature_names=columns)
<<<<<<< HEAD
test_df = pd.read_csv('test/florida_test.csv')
features2 = test_df[list(columns)].values
features2 = np.array(features2)
# print(features2.shape)
importance = clf.feature_importances_
for i, o in zip(columns, importance):
    if o > 0.0:
        print(str(i) + ": " + str(round(o, 2) * 100) + "%")
prediction = clf.predict(features2)
if str(prediction[0]) == "D":
    prediction = "Democrat"
elif str(prediction[0]) == "R":
    prediction = "Republican"
else:
    raise SystemError("Classification Model must be equal to either Democrat or Republican")

print("-------------------------------")
print(str(test_df['State'][0]) + " 2016 Prediction: " + str(prediction))
=======

test_df = pd.read_csv('florida/florida_test2.csv')
print(test_df)
test_columns = ["Average of last 3 elections", "Average of last 5 elections",
                "Average of polls 1 mo before election (>0 = Repub)", "% of registered republicans",
                "% of registered democrats", "State unemployment rate", "Party of governers"]
features2 = test_df[list(columns)].values
features2 = np.array(features2)
# print(features2.shape)
prediction = clf.predict(features2)
print("-------------------------------")
print("2016 Prediction: " + str(prediction))
>>>>>>> 6b7f67e5c85747354cf8c8924176d441a10ccfcc
