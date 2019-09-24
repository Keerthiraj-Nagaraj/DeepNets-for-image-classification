# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:57:33 2018

@author: keert
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

import scipy.io
import time
import pandas as pd
## Keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.losses import categorical_crossentropy

def plot_model_history(model_history, l):
    plt.plot(range(1,len(model_history.history['loss'])+1), model_history.history['loss'])
    plt.title('Learning Curve for ' + l + " PEs in first hidden layer")
    plt.xlabel('Iter #')
    plt.ylabel('Mean Square Error (MSE)')
    plt.grid(True)
    figname = "topo_"+ l +".png"
    plt.savefig(figname)
    plt.show()

data = scipy.io.loadmat('fashion_mnist.mat')

x_train = data['train_x']
y_train = data['train_y']

x_test = data['test_x']
y_test = data['test_y']

x_full = np.concatenate((x_train,x_test),axis=0)
y_full = np.concatenate((y_train,y_test),axis=0)

X=np.concatenate((x_full, y_full), axis=1)

np.random.shuffle(X)
sep=round(70000*2/3)

x_full=X[:,0:784]
y_full=X[:,784:]

x_train = x_full[0:sep]
y_train = y_full[0:sep]

x_test = x_full[sep:]
y_test = y_full[sep:]

h1_pe = 50

num_epochs = 10

lr = [0.1, 0.01, 0.005, 0.001, 0.0005 ] #Number of proceesing elements

rate = 0.005


epochs = 10
act = 'relu'

mse_list = []
acc_list = []
time_list = []

len_vec = lr #this variable is to save time - no need to change variable names
#in results part.... 

loss_plots = np.zeros((epochs, (len(len_vec))))

print("Starting Topology experiments")

t1 = time.time()

x_train_reshaped = x_train.reshape(46667,28,28,1)
x_test_reshaped = x_test.reshape(23333,28,28,1)


input_sha = (28, 28, 1)
num_classes = 10

# create model

print("Starting Topology experiments")

for i, l in enumerate(lr):
    
    print("running for "+ str(l) + " Learning Rate")
    t1 = time.time()
    
    
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape= input_sha))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss= categorical_crossentropy,
                  optimizer= optimizers.SGD(lr=l),
                  metrics=['mse','accuracy']
                  )
    
    model_info = model.fit(x_train_reshaped, y_train, validation_split= 0.1, epochs=10, verbose=2)


    plt.figure()
    plot_model_history(model_info, str(l))
    
    loss_plots[:,i] = model_info.history['loss']
    
    training_scores = model.evaluate(x_test_reshaped, y_test)

    print("For " + str(l) + " PEs")
    print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2]*100))
    
    t2 = time.time()
    
    mse_list.append(training_scores[1]*100)
    acc_list.append(training_scores[2]*100)
    
    time_dur = t2 - t1
    
    time_list.append(time_dur)
    

#Saving results to csv files to plot them

#Hidden layer units
#h1_arr = np.array(num_pe).reshape(len(len_vec),1)

#Learning rate
#lr = [0.005] * len(len_vec)
lr_arr = np.array(lr).reshape(len(len_vec),1)

#Activation function
act_func = [act] * len(len_vec)
act_arr = np.array(act_func).reshape(len(len_vec),1)


#Optimizer

opt = ['SGD'] * len(len_vec)
opt_arr = np.array(opt).reshape(len(len_vec),1)

#Number of epcohs
epoc = [10] * len(len_vec)
epoch_arr = np.array(epoc).reshape(len(len_vec),1)

#MSE, Accuracy and Time for each experiment (in this file, for different # of PEs)
mse_arr = np.array(mse_list).reshape(len(len_vec),1)
acc_arr = np.array(acc_list).reshape(len(len_vec),1)
time_arr = np.array(time_list).reshape(len(len_vec),1)

results = np.concatenate((act_arr, opt_arr, lr_arr, epoch_arr, mse_arr, acc_arr, time_arr), axis=1)    

result = pd.DataFrame(results)
result.to_csv("lr_exp.csv")

#Saving values for learning curves
error_plots = pd.DataFrame(loss_plots)
error_plots.to_csv("lr_error_plots.csv")



