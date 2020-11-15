'''
Created on Oct 27, 2020

@author: david
'''
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
from keras.utils import np_utils
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
import matplotlib.pyplot as plt

from Experiment import Experiment

# datasets
import keras
import keras.datasets as kds
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

# set experiment parameters
trainSize = 50000
testSize = 10000
trainX, trainY, validX, validY, testX, testY = None, None, None, None, None, None

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, baseline=0.4, min_delta=0.001, verbose=1)
mc = ModelCheckpoint('/home/david/diplomska/models/cnn_cifar10.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# define model
def define_model(l2_weight=1e-4, act_fun='relu', dropout=0.2, lr=1e-3, decay=1e-6):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=act_fun, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(l2_weight), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation=act_fun, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(l2_weight)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation=act_fun, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(l2_weight)))
    model.add(Conv2D(64, (3, 3), activation=act_fun, kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(l2_weight)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation=act_fun, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_weight)))
    model.add(Dense(10, activation='softmax'))
    opt = RMSprop(lr=lr, decay=decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# define the function used to evaluate a given configuration
def evaluate_model(params):
    model = define_model(l2_weight=params[0], act_fun=params[1], dropout=params[2], lr=params[3], decay=params[4])
    model.fit_generator(datagen.flow(trainX, trainY, batch_size=params[6]), \
                        steps_per_epoch=trainX.shape[0] // params[6],epochs=params[5],\
                        verbose=1,validation_data=(validX,validY), callbacks=[es, mc])
    estimate = model.evaluate(testX, testY, batch_size=params[6], verbose=1)
    return 1.0 - estimate # convert from a maximizing score to a minimizing score

#z-score
def prep_data(trainX, trainY, testX, testY):
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    mean = np.mean(trainX,axis=(0,1,2,3))
    std = np.std(trainX,axis=(0,1,2,3))
    trainX = (trainX-mean)/(std+1e-7)
    testX = (testX-mean)/(std+1e-7)
    trainY = np_utils.to_categorical(trainY,10)
    testY = np_utils.to_categorical(testY,10)
    return (trainX,trainY), (testX,testY)

if __name__ == '__main__':
    # define the space of hyperparameters to search
    search_space = list()
    search_space.append(Real(1e-6, 1e-2, 'log-uniform', name='l2'))
    search_space.append(Categorical(['relu', 'elu'], name='act_fun'))
    search_space.append(Real(0.1, 0.6, 'uniform', name='dropout'))
    search_space.append(Real(1e-4, 1e-2, 'log-uniform', name='lr'))
    search_space.append(Real(1e-6, 1e-4, 'log-uniform', name='decay'))
    search_space.append(Integer(10, 50, 'log-uniform', name='epochs'))
    search_space.append(Categorical([32,64,128,256], name='batch_size'))
    (trainX, trainY), (testX, testY) = kds.cifar10.load_data()
    (trainY, trainY), (testX, testY) = prep_data(trainX, trainY, testX, testY)
    model = define_model()

    for i in range(3):
        experiment = Experiment(evaluate_model, search_space, numberOfEpochs=80, numberOfRepetitions=1, numberOfRandom=20)
        experiment.run()
        experiment.plot_convergence()

    #===========================================================================
    # # Viri in literatura
    # # https://github.com/mok232/CIFAR-10-Image-Classification
    # # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    #===========================================================================

