# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:57:33 2018

@author: keert
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import scipy.io
import time

## Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def plot_model_history(model_history, pe):
    plt.plot(range(1,len(model_history.history['loss'])+1), model_history.history['loss'])
    plt.title('Learning Rate for a topology with 1 hidden layer of ' + str(pe) + 'PEs')
    plt.xlabel('Iter #')
    plt.ylabel('Mean Square Error (MSE)')
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

num_pe = [10, 20, 50, 75, 100]

mse_list = []
acc_list = []
time_list = []

print("Starting Topology experiments")

for pe in num_pe:
    
    print("running for "+ str(pe) + " PEs")
    t1 = time.time()
    
    # create model
    model = Sequential()
    model.add(Dense(h1_pe, input_dim=784, activation='relu'))
    model.add(Dense(pe, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse','accuracy'])
    
    # Fit the model
    model_info = model.fit(x_train, y_train, epochs=10, batch_size=1)
    
    plt.figure()
    plot_model_history(model_info, pe)
    
    training_scores = model.evaluate(x_test, y_test)
    
    print("For 1 hidden layer with " + str(pe) + " PEs")
    print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2]*100))
    
    t2 = time.time()
    
    mse_list.append(training_scores[1]*100)
    acc_list.append(training_scores[2]*100)
    
    time_dur = t2 - t1
    
    time_list.append(time_dur)
    
    
#h1_arr = np.array(num_pe).reshape(len(num_pe),1)
h1 = [h1_pe] * len(num_pe)

h1_arr = np.array(h1).reshape(len(num_pe),1)
h2_arr = np.array(num_pe).reshape(len(num_pe),1)

mse_arr = np.array(mse_list).reshape(len(num_pe),1)
acc_arr = np.array(acc_list).reshape(len(num_pe),1)
time_arr = np.array(time_list).reshape(len(num_pe),1)

results = np.concatenate((h1_arr, h2_arr, mse_arr, acc_arr, time_arr), axis=1)    

result = pd.DataFrame(results)
#result.to_csv("2_HL_topo_exp.csv")





