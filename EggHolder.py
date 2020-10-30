'''
Created on Oct 20, 2020

@author: david

Compare bayesian optimization on an eggholder function.
Number of epochs: 50
Number of experiments: 100
Return:
    - (mean, variance) values for each epoch
    - time logs for each epoch
    - graph representing function space
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
    
def eggholder_function(x):
    """Returns eggholder function. Minimum -959.6407 at (512,404.23)"""
    return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[0]/2+(x[1]+47))))-x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47))))
    
if __name__ == '__main__':
    search_space = list()
    search_space.append(Real(-513,513, 'uniform', name='x'))
    search_space.append(Real(-513,513, 'uniform', name='y'))
    experiment = Experiment(1, eggholder_function, search_space, numberOfEpochs=10, numberOfRepetitions=1, numberOfRandom=10)
    experiment.run('EI')
    experiment.plot_convergence()
    plt.show()
