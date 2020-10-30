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
from time import time

# datasets
import keras.datasets as kds

#save data
import csv

class Experiment:
    def __init__(self, objectiveFunction, searchSpace, id=None, dir='E:/EXPERIMENTS',
                  numberOfEpochs=50, numberOfRepetitions=10, numberOfRandom=10, seed=None):
        """Set function for optimization."""
        self.dir = dir
        if id == None:
            self.id = self.getid()
        else:
            self.id = id
        self.obj = objectiveFunction
        self.searchSpace = searchSpace
        self.niter = numberOfRepetitions
        self.ncalls = numberOfEpochs
        self.nrand = numberOfRandom
        self.results = []
        self.baseline = []
        self.optimizators = []
        self.savefile = None
        
    def __str__(self):
        EXPERIMENT = """Experiment: {id}
        function: {function}
        search space: {space}
        number of experiments: {n_iter}
        number of epochs per experiment: {n_epoch}
        number of initialization points: {n_init}
        """
        return EXPERIMENT.format(id=self.id, function=self.obj, space=self.searchSpace,
                                  n_iter=self.niter, n_epoch=self.ncalls, 
                                  n_init=self.nrand)
        
    def getid(self):
        id = None
        with open(self.dir + '/id.txt', mode='r+') as id_file:
            id = int(next(id_file))
            id_file.seek(0)
            id_file.write(str(id+1))
            id_file.close()
        return id
        
    def savePoint(self, res):
        timeTaken = time() - self.startime
        csv_writer = csv.writer(self.savefile, delimiter=',')
        csv_writer.writerow(res.x + [res.func_vals[-1].item()] + [timeTaken])
        self.savefile.flush()
    
    def getBaseline(self):
        for i in range(self.niter):
            self.savefile = open(self.dir + '/' + str(self.id) + '/baseline_'+ str(i+1) +'.txt', mode='w')
            result = dummy_minimize(self.obj, self.searchSpace, verbose=True, n_calls=self.ncalls,
                                    callback=[self.savePoint])
            self.baseline.append(result)
            self.savefile.close()
            
    def runExperiment(self,acqFun):
        for i in range(self.niter):
            self.savefile = open(self.dir + '/' + str(self.id) + '/experiment_'+ str(i+1) +'.txt', mode='w')
            randomPoints = self.baseline[i]
            result = gp_minimize(self.obj, self.searchSpace, verbose=True, n_calls=self.ncalls,
                                  n_random_starts=0, acq_func=acqFun,
                                  x0=randomPoints.x_iters[:self.nrand],
                                  y0=randomPoints.func_vals[:self.nrand],
                                  callback=[self.savePoint])
            self.results.append(result)
            self.savefile.close()
        
    def run(self, acqFun):
        os.makedirs(self.dir + '/' + str(self.id))
        with open(self.dir + '/' + str(self.id) + "/description.txt", "w") as experiment_file:
            experiment_file.write(self.__str__())
            experiment_file.close()
        self.startime = time()
        self.getBaseline()
        self.runExperiment(acqFun)
        
    
    def addTuner(self, tuner):
        self.optimizators.append(tuner)
        
    def map(self):
        # create mesh evaluate plot at mesh
        pass
    
    def mean_var(self, points):
        mean = np.zeros(self.ncalls)
        var = np.zeros(self.ncalls)
        for result in points:
            min = float("inf")
            for i in range(self.ncalls):
                if min > result.func_vals[i]:
                    min = result.func_vals[i]
                mean[i] += min
                var[i] += min**2
        mean = mean / self.niter
        var = np.sqrt(var / self.niter - mean**2)
        return (mean, var)
    
    def plot_map(self):
        pass
    
    def plot_convergence(self, file=None):
        if file == None:
            meanvar_results = self.mean_var(self.results)
            meanvar_baseline = self.mean_var(self.baseline)
            plt.plot(range(self.ncalls), meanvar_results[0], 'r', label='bayes optimization')
            plt.fill_between(range(self.ncalls), meanvar_results[0] - meanvar_results[1],
                              meanvar_results[0] + meanvar_results[1], color='red', alpha=0.2)
            plt.plot(range(self.ncalls), meanvar_baseline[0], 'b', label='random search')
            plt.fill_between(range(self.ncalls), meanvar_baseline[0] - meanvar_baseline[1],
                              meanvar_baseline[0] + meanvar_baseline[1], color='blue', alpha=0.2)
            #TODO: plot gridSearch
            plt.legend()
            plt.savefig(self.dir + '/' + str(self.id) + '/convergence'+ str(self.id) +'.png')
    
    def plot_distribution(self):
        pass
    
    def plot_histograms(self):
        pass