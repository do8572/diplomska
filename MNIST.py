'''
Created on Oct 23, 2020

@author: david
'''

import numpy as np
import math
from numpy import mean, var
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, dummy_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import seaborn as sns
import os

from Experiment import Experiment

# datasets
import keras.datasets as kds

# set experiment parameters
trainSize = 10000
testSize = 5000
X, y = None, None

# define the function used to evaluate a given configuration
def evaluate_model(params):
    model = SVC(C=params[0], kernel=params[1], degree=params[2],
                 gamma=params[3], probability=False)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1) # define test harness
    result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy') # calculate 5-fold cross validation
    estimate = mean(result) # calculate the mean of the scores
    return 1.0 - estimate # convert from a maximizing score to a minimizing score


if __name__ == '__main__':
    # define the space of hyperparameters to search
    search_space = list()
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
    search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    search_space.append(Integer(1, 5, name='degree'))
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))
    # load dataset
    (trainX, trainY), (testX, testY) = kds.mnist.load_data()
    trainX, trainY = trainX[0:trainSize].reshape(trainSize, 28*28), trainY[0:trainSize]
    testX, testY = testX[0:testSize].reshape(testSize, 28*28), testY[0:testSize]
    X, y = trainX, trainY
    experiment = Experiment(evaluate_model, search_space, numberOfEpochs=10, numberOfRepetitions=3, numberOfRandom=10)
    experiment.run('EI')
    experiment.plot_convergence()
    plt.show()