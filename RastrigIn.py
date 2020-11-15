'''
Created on Oct 20, 2020

@author: david

Compare bayesian optimization on an eggholder function.
Number of epochs: 200
Number of experiments: 10
Return:
    - (mean, variance) values for each epoch
    - time logs for each epoch
    - graph representing function space
'''

import math
from skopt.space import Real
import matplotlib.pyplot as plt
from time import time

from Experiment import Experiment

dims = [2,4,8]
difs = [1,10,100]
nInits = [10,20,30]
nEpochs = 200
nRepeat = 10 

class RastrigInExperiment:
    def __init__(self, difs=difs, dims=dims, nInits=nInits, nEpochs=nEpochs, nRepeat=nRepeat):
        self.difs = difs
        self.A = difs[0]
        self.dims = dims
        self.nInits = nInits
        self.nEpochs = nEpochs
        self.nRepeat = nRepeat
    
    def rastrigan_function(self, x):
        """Rastrigan function. Minimum 0 at (0,0,...,0)""" 
        rsum = 0
        for xi in x:
            rsum += self.A + xi**2 - self.A*math.cos(2*math.pi*xi)
        return rsum

    def run(self):
        for dim in self.dims:
            search_space = list()
            for i in range(dim):
                search_space.append(Real(-5.12,5.12, 'uniform', name='x' + str(i)))
            for dif in self.difs:
                self.A = dif
                for nInit in self.nInits:
                    experiment = Experiment(self.rastrigan_function, search_space,
                                             numberOfEpochs=self.nEpochs, numberOfRepetitions=self.nRepeat,
                                             numberOfRandom=nInit)
                    experiment.run()
                    experiment.plot_convergence()
                    plt.close()
    
    def analyze(self, eid, dim=2):
        search_space = list()
        for i in range(dim):
            search_space.append(Real(-5.12,5.12, 'uniform', name='x' + str(i)))
        experiment = Experiment(self.rastrigan_function, search_space,
                                             numberOfEpochs=self.nEpochs, numberOfRepetitions=self.nRepeat,
                                             numberOfRandom=nInit)
        experiment.load_results(id=eid)
        #experiment.plot_convergence()
        experiment.plot_convergence_time()
        
    
    
if __name__ == '__main__':
    start =time()
    experiment = RastrigInExperiment()
    experiment.run()
    #experiment.analyze(42, 4)
    end = time()
    print("Runtime: ", end-start)
