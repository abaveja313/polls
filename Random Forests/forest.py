from __future__ import print_function

__author__ = "Amrit Baveja"

import numpy as np
import pandas as pd
from os import system
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

train_df = pd.read_csv('florida/florida2.csv')
# print(train_df)
y = targets = labels = train_df['Results of Last election'].values
y = np.array(y)
columns = ["Average of last 3 elections", "Average of last 5 elections",
           "Average of polls 1 mo before election (>0 = Repub)", "% of registered republicans",
           "% of registered democrats", "State unemployment rate", "Party of governers"]

features = train_df[list(columns)].values
features = np.array(features)
print("Y data \n" + str(y))
print("------------------------")
print("X data \n" + str(features))
print("-------------------------")
X = features
clf = ExtraTreesClassifier(n_estimators=100)
clf2 = DecisionTreeClassifier(max_features=3, criterion="entropy")
clf3 = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
print("X shape: " + str(X.shape))
print("Y shape: " + str(y.shape))
print("--------------------------")
test_df = pd.read_csv('florida/florida_test2.csv')
#print(test_df)
test_columns = ["Average of last 3 elections", "Average of last 5 elections",
                "Average of polls 1 mo before election (>0 = Repub)", "% of registered republicans",
                "% of registered democrats", "State unemployment rate", "Party of governers"]
features2 = test_df[list(columns)].values
features2 = np.array(features2)
# print(features2.shape)
score = clf.score(X,y)
score2 = clf2.score(X,y)
score3 = clf3.score(X,y)
print("Extra Trees Accuracy " + str(score * 100) + "%")
print("Decision Tree Accuracy " + str(score2 * 100) + "%")
print("Random Forest Accuracy " + str(score3 * 100) + "%")
extra_forest = clf.predict(features2)
decision_tree = clf2.predict(features2)
random_forest = clf3.predict(features2)
''''
print("Extra Trees Result " + str(extra_forest))
print("Random Forest Result " + str(random_forest))
print("Decision Tree Result " + str(decision_tree))
'''

outputs = []
def visualize_tree():
    export_graphviz(clf2, out_file='tree.dot', feature_names=columns)
print("---------------------------")

def prediction1(end):
    output = []
    for i in range(0,end):
        predictions = clf.predict(features2)
        if str(predictions) == "[0]":
            output.append(0)
        elif str(predictions) == "[1]":
            output.append(1)
        else:
            raise SystemError
    result = np.mean(output)
    if result > 0.5:
        orig = 1
        winner = "Republican"
    elif result < 0.5:
        orig = 0
        winner = "Democrat"
    else:
        winner = "Tie"
    return winner, output, orig
winner, output, orig = prediction1(30)
print("Extra Trees Result: " + winner)
outputs.append(orig)

def prediction2(end):
    output = []
    for i in range(0,end):
        predictions = clf2.predict(features2)
        if str(predictions) == "[0]":
            output.append(0)
        elif str(predictions) == "[1]":
            output.append(1)
        else:
            raise SystemError
    result = np.mean(output)
    if result > 0.5:
        orig2 = 1
        winner = "Republican"
    elif result < 0.5:
        orig2 = 0
        winner = "Democrat"
    else:
        winner = "Tie"
    return winner, output, orig2
winner2, output2, orig2 = prediction2(30)
print("Decision Tree Result: " + winner2)
outputs.append(orig2)


def prediction3(end):
    output = []
    for i in range(0,end):
        predictions = clf3.predict(features2)
        if str(predictions) == "[0]":
            output.append(0)
        elif str(predictions) == "[1]":
            output.append(1)
        else:
            raise SystemError
    result = np.mean(output)
    if result > 0.5:
        orig3 = 1
        winner = "Republican"
    elif result < 0.5:
        orig3 = 0
        winner = "Democrat"
    else:
        winner = "Tie"
    return winner, output, orig3
winner3, output3, orig3 = prediction3(30)
print("Random Forest Result: " + winner3)
result = None
outputs.append(orig3)
print(outputs)
dcounter = 0
rcounter = 0

for d in outputs:
    if d == 0:
        dcounter +=1
    elif d == 1:
        rcounter +=1
if dcounter > rcounter:
    result = "Democrat"
elif dcounter < rcounter:
    result = "Republican"
else:
    raise SystemError("sklearn module failed to run")
system("open /Users/Amrit/Desktop/polls/Random\ Forests/tree.dot")
#print(output)
visualize_tree()
print("-------------------------------")
print("Florida 2016 Prediction: " + result)
