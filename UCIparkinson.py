'''
Created on Nov 29, 2020

@author: david
'''
import csv
import math
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

import os.path
import matplotlib.pyplot as plt

from Experiment import Experiment
from sklearn.ensemble._gb import GradientBoostingRegressor

def load_data(filename):
    X,y = [],[]
    with open(filename) as data_file:
        data_reader = csv.reader(data_file, delimiter = ',')
        isHeader = True
        for row in data_reader:
            if not isHeader:
                X.append(row[1:4] + row[6:])
                y.append(row[4])
            else:
                isHeader = False
        data_file.close()
    X = np.array(X)
    y = np.array(y)
    return X.astype(np.float),y.astype(np.float)

X,y = load_data('datasets/parkinsons.data')

def max_array(arr):
    new_arr = []
    min = float('inf')
    for i in range(len(arr)):
        if min > arr[i]:
            min = arr[i]
        new_arr.append(min)
    return np.array(new_arr)

def monte_carlo_grid_search(means, stds, n_repeats):
    avg_means = np.zeros(len(means))
    avg_stds = np.zeros(len(means))
    
    for _ in range(n_repeats):
        shuffler = np.random.permutation(len(means))
        new_means = max_array(1 - means[shuffler])
        new_stds = stds[shuffler]
        avg_means = new_means + avg_means
        avg_stds = new_stds + avg_stds
    avg_means /= n_repeats
    avg_stds /= n_repeats
    return avg_means, avg_stds

def evaluate_model(params):
    model = GradientBoostingRegressor(n_estimators=params[0], subsample=params[1],
                                        learning_rate=params[2], max_depth=params[3])
    n_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    if np.isnan(n_scores).any():
        return 200
    return -np.mean(n_scores)

def init_space():
    search_space = list()
    search_space.append(Integer(10, 5000, 'log-uniform', name='n_estimators'))
    search_space.append(Real(0.1, 1.0, 'uniform', name='subsample'))
    search_space.append(Real(0.0001, 1.0, 'log-uniform', name='learning_rate'))
    search_space.append(Integer(4, 10, 'uniform', name='max_depth'))
    return search_space

def save_results(params,means,stds, tarfile):
    with open(tarfile, mode='w', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=',')
        for mean, stdev, param in zip(means, stds, params):
            result_writer.writerow([param, mean, stdev])

if  __name__ == '__main__':
    search_space = init_space()
    for i in range(5):
        experiment = Experiment(evaluate_model, search_space, numberOfEpochs=180, numberOfRepetitions=1, numberOfRandom=10)
        experiment.run(['EI'])
        experiment.plot_convergence()
        plt.close()
