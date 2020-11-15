'''
Created on Oct 23, 2020

@author: david
'''

import numpy as np
import skopt
from skopt import gp_minimize, dummy_minimize
import matplotlib.pyplot as plt
import os
from time import time

#save data
import csv

class ResultData:
    def __init__(self, x_iters, fun_vals, time_vals):
        self.x_iters = x_iters
        self.func_vals = fun_vals
        self.time_vals = time_vals

class Experiment:
    def __init__(self, objectiveFunction, searchSpace, id=None, dir='/home/ocepek/diplomska/EXPERIMENTS',
                  numberOfEpochs=50, numberOfRepetitions=10, numberOfRandom=10, seed=None):
        """Set function for optimization."""
        self.dir = dir
        self.id = id
        self.obj = objectiveFunction
        self.searchSpace = searchSpace
        self.niter = numberOfRepetitions
        self.ncalls = numberOfEpochs
        self.nrand = numberOfRandom
        self.ndims = len(searchSpace)
        self.results_EI = []
        self.time_EI = []
        self.results_PI = []
        self.time_PI = []
        self.results_LCB = []
        self.time_LCB = []
        self.baseline = []
        self.time_baseline = []
        self.optimizators = []
        self.patience = 20
        self.savefile = None
        self.ignorePoint = False
        
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
        with open(self.dir + '/id.txt', mode='r+') as id_file:
            id = int(next(id_file))
            id_file.seek(0)
            id_file.write(str(id+1))
            id_file.close()
        return id
        
    def savePoint(self, res):
        if not self.ignorePoint:
            csv_writer = csv.writer(self.savefile, delimiter=',')
            csv_writer.writerow(res.x_iters[-1] + [res.func_vals[-1].item()] + [time() - self.startime + self.faketime])
            self.savefile.flush()
        else:
            self.ignorePoint = False
        
    def earlyStopping(self, res):
        if np.where(res.func_vals == res.fun)[0][-1] < len(res.func_vals) - self.patience:
            return True
    
    def getBaseline(self):
        self.faketime = 0
        for i in range(self.niter):
            filename = self.dir + '/' + str(self.id) + '/baseline_'+ str(i+1) +'.txt'
            if os.path.isfile(filename):
                x_iters,y_vals,time_vals = self.load_baseline(self.id, i)
                if self.ncalls-len(y_vals) > 0:
                    self.savefile = open(filename, mode='a', newline='')
                    self.startime = time()
                    self.faketime = time_vals[-1]
                    result = dummy_minimize(self.obj, self.searchSpace, verbose=True, n_calls=self.ncalls-len(y_vals),
                                        x0=x_iters, y0=y_vals, callback=[self.savePoint, self.earlyStopping])
                else:
                    result = ResultData(x_iters, y_vals, time_vals)
                self.ignorePoint = True
            else:
                self.savefile = open(filename, mode='w', newline='')
                self.startime = time()
                result = dummy_minimize(self.obj, self.searchSpace, verbose=True, n_calls=self.ncalls,
                                    callback=[self.savePoint, self.earlyStopping])
            self.baseline.append(result)
            if self.savefile != None:
                self.savefile.close()
            
    def runExperiment(self,acqFun):
        for i in range(self.niter):
            filename = self.dir + '/' + str(self.id) + '/'+ acqFun +'_'+ str(i+1) +'.txt'
            self.savefile = open(filename, mode='w', newline='')
            randomPoints = self.baseline[i]
            self.startime = time()
            result = gp_minimize(self.obj, self.searchSpace, verbose=True, n_calls=self.ncalls-self.nrand,
                                  n_random_starts=0, acq_func=acqFun,
                                  x0=randomPoints.x_iters[:self.nrand],
                                  y0=randomPoints.func_vals[:self.nrand],
                                  callback=[self.savePoint])
            if acqFun == 'EI':
                self.results_EI.append(result)
            elif acqFun == 'PI':
                self.results_PI.append(result)
            elif acqFun == 'LCB':
                self.results_LCB.append(result)
            self.savefile.close()
        
    def run(self, acqFun=['EI', 'PI', 'LCB']):
        if self.id == None:
            self.id = self.getid()
        if not os.path.isdir(self.dir + '/' + str(self.id)):
            os.makedirs(self.dir + '/' + str(self.id))
<<<<<<< HEAD
        with open(self.dir + '/' + str(self.id) + "/description.txt", "w") as experiment_file:
            experiment_file.write(self.__str__())
            experiment_file.close()
=======
        if not os.path.isfile(self.dir + '/' + str(self.id) + "/description.txt"):
            with open(self.dir + '/' + str(self.id) + "/description.txt", "w") as experiment_file:
                experiment_file.write(self.__str__())
                experiment_file.close()
>>>>>>> 68a0ecb27e87882213b7ef351054d21d35f321cb
        self.getBaseline()
        for fun in acqFun:
            self.runExperiment(fun)
    
    def addTuner(self, tuner):
        self.optimizators.append(tuner)
        
    def map(self):
        pass
    
    def mean_var(self, points):
        mean = np.zeros(self.ncalls)
        var = np.zeros(self.ncalls)
        for result in points:
            if hasattr(result, 'func_vals'):
                result = result.func_vals
            min = float("inf")
            for i in range(self.ncalls):
                if len(result) > i and min > result[i]:
                    min = result[i]
                mean[i] += min
                var[i] += min**2
        mean = mean / self.niter
        var = np.sqrt(var / self.niter - mean**2)
        return (mean, var)
    
    def plot_map(self):
        pass
    
    def load_baseline(self, id, i):
        fun_file = open(self.dir + '/' + str(id) + '/baseline_'+ str(i+1) +'.txt', mode='r')
        csv_reader = csv.reader(fun_file, delimiter=',')
        x_iters = []
        fun_vals = []
        time_vals = []
        for row in csv_reader:
            x_iter = []
            for j in range(len(row[:-2])):
                if isinstance(self.searchSpace[j], skopt.space.space.Real):
                    x_iter.append(float(row[j]))
                elif isinstance(self.searchSpace[j], skopt.space.space.Integer):
                    x_iter.append(int(row[j]))
                elif isinstance(self.searchSpace[j], skopt.space.space.Categorical):
                    if row[j].isnumeric():
                        x_iter.append(int(row[j]))
                    else:
                        x_iter.append(row[j])
            x_iters.append(x_iter)
            fun_vals.append(float(row[-2]))
            time_vals.append(float(row[-1]))
        fun_file.close()
        return x_iters, fun_vals, time_vals
        
    
    def load_results(self, id=None, acqFuns=['baseline', 'EI', 'PI', 'LCB']):
        self.id = id
        for fun in acqFuns:
            results = []
            for i in range(self.niter):
                fun_file = open(self.dir + '/' + str(id) + '/'+ fun +'_'+ str(i+1) +'.txt', mode='r')
                csv_reader = csv.reader(fun_file, delimiter=',')
                result = []
                if fun != 'baseline':
                    result.extend(self.baseline[i][:self.nrand-1])
                for row in csv_reader:
                    result.append(float(row[self.ndims]))
                results.append(result)
                if fun == 'EI':
                    self.results_EI = results
                elif fun == 'PI':
                    self.results_PI = results
                elif fun == 'LCB':
                    self.results_LCB = results
                elif fun == 'baseline':
                    self.baseline = results          
                fun_file.close()
    
    def plot_convergence(self):
        #TODO: plot gridSearch
        if len(self.results_EI) != 0:
            EI_results = self.mean_var(self.results_EI)
            plt.plot(range(self.ncalls), EI_results[0], 'b', label='EI')
            plt.fill_between(range(self.ncalls), EI_results[0] - EI_results[1],
                                  EI_results[0] + EI_results[1], color='blue', alpha=0.2)
        if len(self.results_PI) != 0:
            PI_results = self.mean_var(self.results_PI)
            plt.plot(range(self.ncalls), PI_results[0], 'g', label='PI')
            plt.fill_between(range(self.ncalls), PI_results[0] - PI_results[1],
                                  PI_results[0] + PI_results[1], color='green', alpha=0.2)
        if len(self.results_LCB) != 0:
            LCB_results = self.mean_var(self.results_LCB)
            plt.plot(range(self.ncalls), LCB_results[0], 'r', label='LCB')
            plt.fill_between(range(self.ncalls), LCB_results[0] - LCB_results[1],
                                  LCB_results[0] + LCB_results[1], color='red', alpha=0.2)
        meanvar_baseline = self.mean_var(self.baseline)
        plt.plot(range(self.ncalls), meanvar_baseline[0], 'k', label='random search')
        plt.fill_between(range(self.ncalls), meanvar_baseline[0] - meanvar_baseline[1],
                              meanvar_baseline[0] + meanvar_baseline[1], color='black', alpha=0.1)
        plt.legend()
        plt.savefig(self.dir + '/' + str(self.id) + '/convergence'+ str(self.id) +'.png')
    
    def load_time(self, id=None, acqFuns=['baseline', 'EI', 'PI', 'LCB']):
        self.id = id
        for fun in acqFuns:
            results = []
            for i in range(self.niter):
                fun_file = open(self.dir + '/' + str(id) + '/'+ fun +'_'+ str(i+1) +'.txt', mode='r')
                csv_reader = csv.reader(fun_file, delimiter=',')
                result = []
                start = 0
                if fun != 'baseline':
                    result.extend(self.time_baseline[i][:self.nrand-1])
                    start = result[-1]
                for row in csv_reader:
                    result.append(float(row[-1]) + start)
                results.append(result)
            if fun == 'EI':
                self.time_EI = results
            elif fun == 'PI':
                self.time_PI = results
            elif fun == 'LCB':
                self.time_LCB = results
            elif fun == 'baseline':
                self.time_baseline = results         
            fun_file.close()
        
    def plot_convergence_time(self):
        self.load_time(id=self.id)
        if len(self.results_EI) != 0 and len(self.results_EI) == len(self.time_EI):
            EI_results = self.mean_var(self.results_EI)
            EI_times = np.mean(np.array(self.time_EI), axis=0)
        if len(self.results_PI) != 0 and len(self.results_PI) == len(self.time_PI):
            PI_results = self.mean_var(self.results_PI)
            PI_times = np.mean(np.array(self.time_PI), axis=0)
        if len(self.results_LCB) != 0 and len(self.results_LCB) == len(self.time_LCB):
            LCB_results = self.mean_var(self.results_LCB)
            LCB_times = np.mean(np.array(self.time_LCB), axis=0)
        meanvar_baseline = self.mean_var(self.baseline)
        meanvar_time_baseline = np.mean(np.array(self.time_baseline), axis=0)
        plt.plot(EI_times, EI_results[0], 'b', label='EI')
        plt.plot(PI_times, PI_results[0], 'g', label='PI')
        plt.plot(LCB_times, LCB_results[0], 'r', label='LCB')
        plt.plot(meanvar_time_baseline, meanvar_baseline[0], 'k', label='random search')
        plt.legend()
        plt.savefig(self.dir + '/' + str(self.id) + '/convergence'+ str(self.id) +'.png')
    
    def plot_distribution(self):
        pass
    
    def plot_histograms(self):
        pass
