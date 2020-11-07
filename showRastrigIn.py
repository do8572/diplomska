'''
Created on Oct 18, 2020

@author: david
'''

#===============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# 
# bounds = np.array([[-1.0, 2.0]])
# noise = 0.2
# 
# def f(X, noise=noise):
#     return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)
# 
# X_init = np.array([[-0.9], [1.1]])
# Y_init = f(X_init)
# 
# 
# import GPy
# import GPyOpt
# 
# from GPyOpt.methods import BayesianOptimization
# 
# kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
# bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]
# 
# optimizer = BayesianOptimization(f=f, 
#                                  domain=bds,
#                                  model_type='GP',
#                                  kernel=kernel,
#                                  acquisition_type ='EI',
#                                  acquisition_jitter = 0.01,
#                                  X=X_init,
#                                  Y=-Y_init,
#                                  noise_var = noise**2,
#                                  exact_feval=False,
#                                  normalize_Y=False,
#                                  maximize=True)
# 
# optimizer.run_optimization(max_iter=10)
# optimizer.plot_acquisition()
#===============================================================================


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
 
def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])
 
def rastrigan_function(x):
    """Rastrigan function. Minimum 0 at (0,0,...,0)""" 
    A = 100
    n = len(x)
    return A*n + np.sum(x**2 - A*np.cos(2*math.pi*x), axis=0)
 
if __name__ == '__main__':
    X = np.linspace(-5.12, 5.12, 150)    
    Y = np.linspace(-5.12, 5.12, 150)  
  
    X, Y = np.meshgrid(X, Y)
    XY = np.stack([X, Y]) 
  
    Z = rastrigan_function(XY)
  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
  
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)    
    plt.savefig('rastrigin.png')
    plt.show()

