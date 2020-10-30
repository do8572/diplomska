'''
Created on Oct 27, 2020

@author: david
'''
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.svm import SVC
# from skopt.space import Integer
# from skopt.space import Real
# from skopt.space import Categorical
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
X, y = None, None

# define model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.0001), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.0001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001)))
    model.add(Dense(10, activation='softmax'))
    # compile model
    # opt = SGD(lr=0.001, momentum=0.9)
    opt = RMSprop(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    #===========================================================================
    # # save plot to file
    # filename = sys.argv[0].split('/')[-1]
    # plt.savefig(filename + '_plot.png')
    # plt.close()
    #===========================================================================
    plt.show()

#===============================================================================
# # define the function used to evaluate a given configuration
# def evaluate_model(params):
#     model = SVC(C=params[0], kernel=params[1], degree=params[2],
#                  gamma=params[3], probability=False)
#     cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1) # define test harness
#     result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy') # calculate 5-fold cross validation
#     estimate = mean(result) # calculate the mean of the scores
#     return 1.0 - estimate # convert from a maximizing score to a minimizing score
#===============================================================================

def prep_pixels(trainX, testX):
    # convert from integers to floats
    train_norm = trainX.astype('float32')
    test_norm = testX.astype('float32')
    # normalize to range 0-1
    train_norm /= 255.0
    test_norm /= 255.0
    return train_norm, test_norm

def prep_output(trainY, testY):
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainY, testY


if __name__ == '__main__':
    #===========================================================================
    # # define the space of hyperparameters to search
    # search_space = list()
    # search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
    # search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    # search_space.append(Integer(1, 5, name='degree'))
    # search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))
    #===========================================================================
    # load dataset
    (trainX, trainY), (testX, testY) = kds.cifar10.load_data()
    # sumarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(3,3, 1 + i)
        plt.imshow(trainX[i])
    plt.savefig('experiments/cifar_img.png')
    plt.close()
    # one hot encode target values
    trainY, testY = prep_output(trainY, testY)
    # normalize pixels
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # define early stopping
    es = EarlyStopping(monitor='val_acc', mode='max', patience=10, baseline=0.4, min_delta=0.001, verbose=1)
    # define model saving
    mc = ModelCheckpoint('/home/david/diplomska/models/cnn_cifar10.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.2,
                        callbacks=[es, mc], verbose=1)
    # load saved model
    saved_model = load_model('/home/david/diplomska/models/cnn_cifar10.h5')
    # evalueate model
    _, acc = saved_model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)
    # save model
    # pickle.dump(model, open('E:/CIFAR10/model.sav', mode='wb'))
    #===========================================================================
    # trainX, trainY = trainX[0:trainSize].reshape(trainSize, 32*32*3), trainY[0:trainSize]
    # testX, testY = testX[0:testSize].reshape(testSize, 32*32*3), testY[0:testSize]
    # X, y = trainX, trainY
    # experiment = Experiment(evaluate_model, search_space, numberOfEpochs=10, numberOfRepetitions=3, numberOfRandom=10)
    # experiment.run('EI')
    # experiment.plot_convergence()
    # plt.show()
    #===========================================================================

    #===========================================================================
    # # Viri in literatura
    # # https://github.com/mok232/CIFAR-10-Image-Classification
    # # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    #===========================================================================
