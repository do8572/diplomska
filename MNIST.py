'''
Created on Oct 23, 2020

@author: david
'''

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
import matplotlib.pyplot as plt

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

def prep_pixels(train, test):
    train = train.astype('float32')
    test = test.astype('float32')
    train = train / 255.0
    test = test / 255.0
    return train,test

def init_space():
    search_space = list()
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
    search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    search_space.append(Integer(1, 5, name='degree'))
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))
    return search_space


if __name__ == '__main__':
    # define the space of hyperparameters to search
    search_space = init_space()
    # load dataset
    (trainX, trainY), (testX, testY) = kds.mnist.load_data()
    trainX, trainY = trainX[0:trainSize].reshape(trainSize, 28*28), trainY[0:trainSize]
    testX, testY = testX[0:testSize].reshape(testSize, 28*28), testY[0:testSize]
    # scale data
    trainX, testX = prep_pixels(trainX, testX)
    # set data
    X, y = trainX, trainY
    experiment = Experiment(evaluate_model, search_space, numberOfEpochs=100, numberOfRepetitions=3, numberOfRandom=30)
    experiment.run()
    experiment.plot_convergence()
    plt.show()
