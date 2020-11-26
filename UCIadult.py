'''
Created on Nov 24, 2020

@author: david

Experiment: Bayesian optimization on GBoost with Wine dataset
Optimized hyperparameters: Number of trees, Number of samples, Number of features, learning rate, Tree deapth
'''
import csv
import numpy as np
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
from time import time

def vectorize(row):
    
    # imputate missing values
    return row

def load_data(filename):
    X,y = [],[]
    with open(filename) as data_file:
        data_reader = csv.reader(data_file, delimiter = ',')
        for row in data_reader:
            X.append(vectorize(row[:-1]))
            y.append(row[-1])
        data_file.close()
    X = np.array(X)
    y = np.array(y)
    return X,y

X_train,y_train = load_data('datasets/adult.data')
enc = OneHotEncoder(dtype=np.int)

def convert_categorical(X):
    enc.fit(X[:, [1,3,5,6,7,8,9,13]])
    a = enc.transform(X[:,[1,3,5,6,7,8,9,13]]).toarray()
    b = X[:,[0,2,4,10,11,12]]
    X = np.concatenate((a,b), axis=1).astype(np.int)
    #np.savetxt('test.txt', X_train, delimiter=',', fmt='%.0f')
    return X

X_train = convert_categorical(X_train)

def evaluate_model(params):
    model = RandomForestClassifier(n_estimators=params[0], max_samples=params[1],
                                        max_features=params[2])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
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
    search_space.append(Integer(10, 500, 'log-uniform', name='n_estimators'))
    search_space.append(Real(0.1, 1.0, 'uniform', name='max_samples'))
    search_space.append(Integer(4, 16, 'log-uniform', name='max_features'))
    return search_space

if  __name__ == '__main__':    
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['max_samples'] = [0.3, 0.6, 0.9]
    grid['max_features'] = [4,8,10,12,16]
    model = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    start1 = time()
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', verbose=1)
    grid_result = grid_search.fit(X_train, y_train)
    end1 = time()
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    save_results(means,stds,params, 'datasets/adultGS.csv')
    search_space = init_space()
    start2 = time()
    experiment = Experiment(evaluate_model, search_space, numberOfEpochs=108, numberOfRepetitions=10, numberOfRandom=10)
    experiment.run(['EI'])
    end2 = time()
    experiment.plot_convergence()
    #===========================================================================
    # axes = plt.gca()
    # axes.set_ylim([0.01,0.035])
    #===========================================================================
    GS_mean, GS_std = monte_carlo_grid_search(means, stds, 100000)
    plt.plot(range(180), max_array(GS_mean), 'b', label='GS')
    plt.legend()
    plt.show()
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(end1 - start1)
    print(end2 - start2)

#===============================================================================
# Literatura in viri:
# https://en.wikipedia.org/wiki/Gradient_boosting
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
#=============================================================
