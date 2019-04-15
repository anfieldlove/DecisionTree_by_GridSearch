import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def decisionTree(data_name, p, q):
    data = pd.read_csv(data_name, header=None)
    data1 = data.drop([0])
    a = data.shape[1] - 1
    x = data1[data.columns[:-1]]
    y = data1[a]
    x_train, x_test, y_train, y_test = train_test_split(x, y)


    def md_score(depth):
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_socre = model.score(x_train, y_train)
        return test_score, train_socre

    depth = range(p, q)
    score = [md_score(dp) for dp in depth]
    test_s = [s[0] for s in score]
    train_s = [s[1] for s in score]
    plt.plot(depth, test_s)
    plt.plot(depth, train_s)
    plt.show()


def grid_search(data_name, p, q):
    data = pd.read_csv(data_name, header=None)
    data1 = data.drop([0])
    a = data.shape[1] - 1
    x = data1[data.columns[:-1]]
    y = data1[a]

    depth = range(p, q)
    value = np.linspace(0, 0.5)
    params = {'max_depth': depth, 'min_impurity_decrease': value}
    model = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
    model.fit(x, y)
    print(model.best_params_, model.best_score_)
    return model.best_params_, model.best_score_

