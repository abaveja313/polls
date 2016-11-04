import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.colors import rgb2hex
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable as make_axes_locatable

__author__ = "Amrit Baveja"
start_time = time.clock()
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
# draw state boundaries.
# data from U.S Census Bureau
# http://www.census.gov/geo/www/cob/st2000.html
shp_info = m.readshapefile('st99_d00', 'states', drawbounds=True)
probabils = []
republican_votes = 0
democrat_votes = 0
electoral_votes = {
    'New Jersey': 14,
    'Rhode Island': 4,
    'Massachusetts': 11,
    'Connecticut': 7,
    'Maryland': 10,
    'New York': 29,
    'Delaware': 3,
    'Florida': 29,
    'Ohio': 18,
    'Pennsylvania': 20,
    'Illinois': 20,
    'California': 55,
    'Hawaii': 4,
    'Virginia': 13,
    'Michigan': 16,
    'Indiana': 11,
    'North Carolina': 15,
    'Georgia': 16,
    'Tennessee': 11,
    'New Hampshire': 4,
    'South Carolina': 9,
    'Louisiana': 8,
    'Kentucky': 8,
    'Wisconsin': 10,
    'Washington': 12,
    'Alabama': 9,
    'Missouri': 10,
    'Texas': 38,
    'West Virginia': 5,
    'Vermont': 3,
    'Minnesota': 10,
    'Mississippi': 6,
    'Iowa': 6,
    'Arkansas': 6,
    'Oklahoma': 7,
    'Arizona': 11,
    'Colorado': 9,
    'Maine': 4,
    'Oregon': 7,
    'Kansas': 6,
    'Utah': 6,
    'Nebraska': 5,
    'Nevada': 6,
    'Idaho': 4,
    'New Mexico': 5,
    'South Dakota': 3,
    'North Dakota': 3,
    'Montana': 3,
    'Wyoming': 3,
    'District of Columbia': 3,
    'Alaska': 3
}

popdensity = {
    'New Jersey': 0.0,
    'Rhode Island': 0.0,
    'Massachusetts': 0.0,
    'Connecticut': 0.0,
    'Maryland': 0.0,
    'New York': 0.0,
    'Delaware': 0.0,
    'Florida': 0.0,
    'Ohio': 0.0,
    'Pennsylvania': 0.0,
    'Illinois': 0.0,
    'California': 0.0,
    'Hawaii': 0.0,
    'Virginia': 0.0,
    'Michigan': 0.0,
    'Indiana': 0.0,
    'North Carolina': 0.0,
    'Georgia': 0.0,
    'Tennessee': 0.0,
    'New Hampshire': 0.0,
    'South Carolina': 0.0,
    'Louisiana': 0.0,
    'Kentucky': 0.0,
    'Wisconsin': 0.0,
    'Washington': 0.0,
    'Alabama': 0.0,
    'Missouri': 0.0,
    'Texas': 0.0,
    'West Virginia': 0.0,
    'Vermont': 0.0,
    'Minnesota': 0.0,
    'Mississippi': 0.0,
    'Iowa': 0.0,
    'Arkansas': 0.0,
    'Oklahoma': 0.0,
    'Arizona': 0.0,
    'Colorado': 0.0,
    'Maine': 0.0,
    'Oregon': 0.0,
    'Kansas': 0.0,
    'Utah': 0.0,
    'Nebraska': 0.0,
    'Nevada': 0.0,
    'Idaho': 0.0,
    'New Mexico': 0.0,
    'South Dakota': 0.0,
    'North Dakota': 0.0,
    'Montana': 0.0,
    'Wyoming': 0.0,
    'District of Columbia': 0.0,
    'Alaska': 0.0
}
print(shp_info)
# choose a color for each state based on population density.
colors = {}
ax = plt.gca()
statenames = []
cmap = plt.cm.bwr
# use 'hot' colormap
vmin = 0;
vmax = 1  # set range.

print(m.states_info[0].keys())
states = ['test/alaska_test.csv',
          'test/alabama_test.csv',
          'test/arkansas_test.csv',
          'test/arizona_test.csv',
          'test/california_test.csv',
          'test/colorado_test.csv',
          'test/connecticut_test.csv',
          'test/washington-dc_test.csv',
          'test/delaware_test.csv',
          'test/florida_test.csv',
          'test/georgia_test.csv',
          'test/hawaii_test.csv',
          'test/iowa_test.csv',
          'test/idaho_test.csv',
          'test/illinois_test.csv',
          'test/indiana_test.csv',
          'test/kansas_test.csv',
          'test/kentucky_test.csv',
          'test/louisiana_test.csv',
          'test/maine_test.csv',
          'test/maryland_test.csv',
          'test/massachusetts_test.csv',
          'test/michigan_test.csv',
          'test/minnesota_test.csv',
          'test/mississippi_test.csv',
          'test/missouri_test.csv',
          'test/montana_test.csv',
          'test/nebraska-test.csv',
          'test/nevada_test.csv',
          'test/new-hampshire_test.csv',
          'test/new-jersey_test.csv',
          'test/new-mexico_test.csv',
          'test/new-york_test.csv',
          'test/north-carolina_test.csv',
          'test/north-dakota_test.csv',
          'test/ohio_test.csv',
          'test/oklahoma-test.csv',
          'test/oregon_test.csv',
          'test/pennsylvania_test.csv',
          'test/rhode-island_test.csv',
          'test/south-carolina_test.csv',
          'test/south-dakota_test.csv',
          'test/tennessee_test.csv',
          'test/texas_test.csv',
          'test/utah_test.csv',
          'test/vermont_test.csv',
          'test/virgina_test.csv',
          'test/washington_test.csv',
          'test/west-virginia_test.csv',
          'test/wisconsin_test.csv',
          'test/wyoming_test.csv',
          ]


def open_csv_create(file_path):
    train_df = pd.read_csv(file_path)
    # print(train_df)
    y = train_df['Outcome'].values
    y = np.array(y)
    column = ["Polls", "Last Election", "Unemployment", "ISM"]
    features = train_df[list(column)].values
    features = np.array(features)
    return features, y, column


features, y, columns = open_csv_create('model_new_new.csv')


def print_shape_data():
    print("Y data \n" + str(y))
    print("------------------------")
    X = features
    print("X data \n" + str(X))
    print("-------------------------")

    print("X shape: " + str(X.shape))
    print("Y shape: " + str(y.shape))
    print("--------------------------")
    return X


X = print_shape_data()


def find_feature_importance(x_var, y_var, max_features):
    pred = DecisionTreeClassifier(criterion="entropy", max_features=max_features, splitter="best")
    pred.fit(x_var, y_var)
    importance = pred.feature_importances_
    tree.export_graphviz(pred, out_file="network-tree.dot", feature_names=columns)
    for i, o in zip(columns, importance):
        if o > 0.0:
            print(str(i) + ": " + str(round(o, 2) * 100) + "%")
    return importance


print("Most Important Features:")
x = find_feature_importance(X, y, 2)
print("-------------------------")


def create_neural_network(layerx, layery, csv, verbose):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layerx, layery), random_state=1,
                        verbose=verbose)
    clf.fit(X, y)
    test_df = pd.read_csv(csv)
    # print(test_df)
    test_columns = columns
    features2 = test_df[list(test_columns)].values
    score = clf.score(X, y)
    print("Neural Network Accuracy " + str(round(score, 2) * 100) + "%")
    result = clf.predict(features2)
    if str(result[0]) == "D":
        result = "Democrat"
    else:
        result = "Republican"
    output2 = []
    probab = clf.predict_proba(features2)
    if str(round(probab[0][0], 3) * 100) == "50.0" and str(round(probab[0][1], 3) * 100) == "50.0":
        result = "Tie"

    return test_df, test_columns, features2, clf, result, probab, output2


def loop_results():
    l = 0
    global republican_votes
    global democrat_votes
    for i in states:
        r = None
        test_df, test_columns, features2, clf, result, probability, output2 = create_neural_network(22, 1, i, False)
        print("Probability: \n" + "Democrat: " + str(round(probability[0][0], 3) * 100) + "%" + "\nRepublican: " + str(
            round(probability[0][1], 3) * 100) + "%")
        statei = test_df['State'][0].lower()
        statei2 = statei.title()

        if statei2 == "District Of Columbia":
            statei2 = "District of Columbia"
        votes = int(electoral_votes[statei2])
        if str(result) == "Democrat":
            r = 'blue'
            democrat_votes += votes
        elif str(result) == "Republican":
            r = 'red'
            republican_votes += votes
        print('\033[1m' + statei2 + " 2016 Prediction: " + str(result) + '\033[0m')
        probabils.append(round(probability[0][0], 3))
        popdensity[statei2] = round(probability[0][0], 3)
        print("---------------------")
    print('\033[1m' + "Republican Electoral Votes " + str(republican_votes) + '\033[0m')
    print('\033[1m' + "Democrat Electoral Votes " + '\033[1m' + str(democrat_votes) + '\033[0m')
    print("--------------------------")

loop_results()


def map_helper():
    print(popdensity)
    print("------------------------------------")
    statenames = []
    colors = {}
    cmap = plt.cm.bwr  # use 'hot' colormap
    vmin = 0;
    vmax = 1  # set range.
    print(m.states_info[0].keys())
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia', 'Puerto Rico']:
            pop = popdensity[statename]
            colors[statename] = cmap(1. - np.sqrt((pop - vmin) / (vmax - vmin)))[:3]
        statenames.append(statename)
    return statename, pop, colors, cmap, statenames


statenames, pop, colors, cmap, statenames = map_helper()


# cycle through state names, color each one.
def create_states():
    # get current axes instance
    for nshape, seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)


create_states()

# print(features2.shape)
# plt.annotate(str(mi), xy = (7347280, 4774970), xytext=(6977770,5334230), arrowprops=dict(arrowstyle="<-"))
# plt.annotate(str(mi), xy = (915848, 2182980), xytext=(1318980, 1653130), arrowprops=dict(arrowstyle="<-"))
final_time = time.clock() - start_time
print("-------------------------")
print("Ran in " + str(round(final_time, 2)) + " seconds")
print("Generating Map...")
time.sleep(5)
plt.hold(False)
plt.title("Republican Electoral Votes: " + str(republican_votes) + "\n " + "Democrat Electoral Votes: " + str(democrat_votes))
plt.show()
