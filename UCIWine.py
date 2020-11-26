'''
Created on Nov 24, 2020

@author: david

Experiment: Bayesian optimization on GBoost with Wine dataset
Optimized hyperparameters: Number of trees, Number of samples, Number of features, learning rate, Tree deapth
'''
import csv
import numpy as np
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

import os.path
import matplotlib.pyplot as plt

from Experiment import Experiment

#TODO: one hot encoding
#TODO: Bayesian optimization      

def load_wine():
    X,y = [],[]
    with open('datasets/wine.data') as data_file:
        data_reader = csv.reader(data_file, delimiter = ',')
        for row in data_reader:
            X.append(row[1:])
            y.append(row[0])
        data_file.close()
    X = np.array(X).astype(np.float)
    y = np.array(y).astype(np.integer)
    return X,y

X,y = load_wine()

def evaluate_model(params):
    model = GradientBoostingClassifier(n_estimators=params[0], subsample=params[1],
                                        #max_features=params[2],
                                        learning_rate=params[2], max_depth=params[3])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
    return 1 - np.mean(n_scores)

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

def save_results(params,means,stds, tarfile):
    with open(tarfile, mode='w', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=',')
        for mean, stdev, param in zip(means, stds, params):
            result_writer.writerow([param, mean, stdev])
            
def load_results(tarfile):
    points, means, stds = [], [], []
    with open(tarfile, mode='r') as result_file:
        result_reader = csv.reader(result_file, delimiter=',')
        for row in result_reader:
            points.append(row[2])
            means.append(row[0])
            stds.append(row[1])
        result_file.close()
    return points, np.array(means).astype(np.float), \
        np.array(stds).astype(np.float)

def init_space():
    search_space = list()
    search_space.append(Integer(10, 5000, 'log-uniform', name='n_estimators'))
    search_space.append(Real(0.1, 1.0, 'uniform', name='subsample'))
    #search_space.append(Integer(1, 20, 'uniform', name='max_features'))
    search_space.append(Real(0.0001, 1.0, 'log-uniform', name='learning_rate'))
    search_space.append(Integer(1, 10, 'uniform', name='max_depth'))
    return search_space

if  __name__ == '__main__':
    if os.path.isfile('datasets/wineGS.csv'):
        search_space = init_space()
        experiment = Experiment(evaluate_model, search_space, numberOfEpochs=180, numberOfRepetitions=5, numberOfRandom=10)
        experiment.run(['EI'])
        experiment.plot_convergence()
        #plt.close()
        axes = plt.gca()
        axes.set_ylim([0.01,0.035])
        _, means, stds = load_results('datasets/wineGS.csv')
        GS_mean, GS_std = monte_carlo_grid_search(means, stds, 100000)
        plt.plot(range(180), max_array(GS_mean), 'b', label='GS')
        #=======================================================================
        # plt.fill_between(range(180), max_array(mean) - max_array(std),
        #                           max_array(mean) + max_array(std), color='blue', alpha=0.2)
        #=======================================================================
        plt.legend()
        plt.show()
    else:
        grid = dict()
        grid['n_estimators'] = [10, 50, 100, 500]
        grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
        grid['subsample'] = [0.5, 0.7, 1.0]
        grid['max_depth'] = [3, 7, 9]
        model = GradientBoostingClassifier()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', verbose=1)
        grid_result = grid_search.fit(X, y)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        save_results(means,stds,params, 'E:/UCIwine/grid_search.csv')
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#===============================================================================
# Literatura in viri:
# https://en.wikipedia.org/wiki/Gradient_boosting
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/
#===============================================================================
