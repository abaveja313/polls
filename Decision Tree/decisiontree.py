from __future__ import print_function

__author__ = "Amrit Baveja"

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

features = ['Average of Last 3 elections', 'Average of last 5 elections', 'Average of polls 1 mo before election',
            '% of registered republicans', '% of registered democrats', 'Unemployment', 'Governor Party']


def get_training_data(file_name):
    data = pd.read_csv(file_name)
    nvalues = {"x1": [], "x2": [], "x3": [], "x4": [], "x5": [], "x6": [], "x7": [], "y": []}
    for i, o in zip(data['Year'], data['Results of Last election']):
        nvalues['y'].append(float(o))
    for t, m in zip(data['Average of last 3 elections'], data['Average of last 5 elections']):
        nvalues['x1'].append(float(t))
        nvalues['x2'].append(float(m))
    for y, z in zip(data['Average of polls 1 mo before election (<0 = Repub)'], data['% of registered republicans']):
        nvalues['x3'].append(float(y))
        nvalues['x4'].append(float(z))
    for a, b, c in zip(data['% of registered democrats'], data['State unemployment rate'], data['Party of governers']):
        nvalues['x5'].append(float(a))
        nvalues['x6'].append(float(b))
        nvalues['x7'].append(float(c))
    # return years, results, avg3, avg5, avg_1mo, perc_demo, perc_repub,unemployed, governer_party
    return nvalues


# year, result, avg3, avg5, avg_1mo, perc_demo, perce_repub, unemployed, governer_party = get_data('/Users/Amrit/Desktop/florida2.csv')
nvalues = get_training_data('/Users/Amrit/Desktop/florida2.csv')
a = np.array([nvalues["x1"], nvalues["x2"], nvalues['x3'], nvalues['x4'], nvalues['x5'], nvalues['x6'], nvalues['x7']])
b = np.array(nvalues["y"])
a = a.reshape(len(nvalues['x1']), 7)
b = b.reshape(len(nvalues["y"]), 1)
print(a)
print(b)
print(a.shape)
print(b.shape)
dtree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=500)
dtree.fit(a, b)
print("Score: " + str(dtree.score(a, b)))


def get_test_data(file_name):
    data = pd.read_csv(file_name)
    nvalues2 = {"x1": [], "x2": [], "x3": [], "x4": [], "x5": [], "x6": [], "x7": [], "y": []}
    for i, o in zip(data['Year'], data['Results of Last election']):
        nvalues2['y'].append(float(o))
    for t, m in zip(data['Average of last 3 elections'], data['Average of last 5 elections']):
        nvalues2['x1'].append(float(t))
        nvalues2['x2'].append(float(m))
    for y, z in zip(data['Average of polls 1 mo before election (<0 = Repub)'], data['% of registered republicans']):
        nvalues2['x3'].append(float(y))
        nvalues2['x4'].append(float(z))
    for a, b, c in zip(data['% of registered democrats'], data['State unemployment rate'], data['Party of governers']):
        nvalues2['x5'].append(float(a))
        nvalues2['x6'].append(float(b))
        nvalues2['x7'].append(float(c))
    # return years, results, avg3, avg5, avg_1mo, perc_demo, perc_repub,unemployed, governer_party
    return nvalues2


nvalues2 = get_training_data('/Users/Amrit/Desktop/florida_test.csv')
print(nvalues2)
a1 = np.array(
    [nvalues2["x1"], nvalues2["x2"], nvalues2['x3'], nvalues2['x4'], nvalues2['x5'], nvalues2['x6'], nvalues2['x7']])
new_a1 = a1.reshape(len(nvalues2['x1']), 7)
pred = dtree.predict(new_a1)
print(pred)


def visualize_tree():
    export_graphviz(dtree, out_file='tree.dot')  # produces dot file


visualize_tree()
