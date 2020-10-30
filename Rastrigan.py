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

import math
from skopt.space import Real
import matplotlib.pyplot as plt

from Experiment import Experiment

def rastrigan_function(x):
    """Rastrigan function. Minimum 0 at (0,0,...,0)""" 
    A = 10
    sum = 0
    for xi in x:
        sum += A + xi**2 - A*math.cos(2*math.pi*xi)
    return sum
    
if __name__ == '__main__':
    search_space = list()
    search_space.append(Real(-5.12,5.12, 'uniform', name='x1'))
    search_space.append(Real(-5.12,5.12, 'uniform', name='x2'))
    search_space.append(Real(-5.12,5.12, 'uniform', name='x3'))
    search_space.append(Real(-5.12,5.12, 'uniform', name='x4'))
    experiment = Experiment(rastrigan_function, search_space, numberOfEpochs=10, numberOfRepetitions=1, numberOfRandom=10)
    experiment.run('EI')
    experiment.plot_convergence()
    plt.show()

