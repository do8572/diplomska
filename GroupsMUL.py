'''
Created on Dec 19, 2020

@author: david
'''
import pickle as pkl
import csv
import numpy as np
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from skopt.space import Integer
from skopt.space import Real

import matplotlib.pyplot as plt

from Experiment import Experiment
from sklearn import preprocessing
from xgboost import XGBClassifier

def load_data():
    with open('datasets/gmul.pkl', 'rb') as fileObject:
        data = pkl.load(fileObject)
        fileObject.close()
        lencoder = preprocessing.LabelEncoder()
        lencoder.fit(data[:,-1])
        return data[:,0:-1], lencoder.transform(data[:,-1])

X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

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
            
def init_space():
    search_space = list()
    search_space.append(Real(0.01, 0.2, 'log-uniform', name='learning_rate'))
    search_space.append(Integer(3, 8, 'uniform', name='max_depth'))
    search_space.append(Integer(1, 9, 'uniform', name='min_child_weight'))
    search_space.append(Real(0.1, 1.0, 'uniform', name='gamma'))
    search_space.append(Real(0.3, 0.9, 'uniform', name='colsample_bytree'))
    return search_space

def evaluate_model(params):
    model = XGBClassifier(use_label_encoder=False, n_estimators=100,
                          learning_rate=params[0], min_child_weight=params[2], max_depth=params[1], gamma=params[3], n_jobs=-1,
                          colsample_bytree =params[4], eval_metric='mlogloss') # subsample=0.9
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv,
                               n_jobs=-1, verbose=1)
    return 1 - mean(n_scores)

if __name__ == '__main__':
    search_space = init_space()
    experiment = Experiment(evaluate_model, search_space, numberOfEpochs=129, numberOfRepetitions=1, numberOfRandom=10)
    experiment.run(['EI'])
    experiment.plot_convergence()
    plt.legend()
    plt.show()
    
#===============================================================================
# # Viri in literatura
# https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning
