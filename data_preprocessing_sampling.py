# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:57:33 2018

@author: keert
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.io
import time

## Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


##SKlearn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def plot_model_history(model_history, l):
    plt.plot(range(1,len(model_history.history['loss'])+1), model_history.history['loss'])
    plt.title('Learning Curve for ' + str(l) + " sample jumps")
    plt.xlabel('Iter #')
    plt.ylabel('Mean Square Error (MSE)')
    plt.grid(True)
    figname = "down_samp_"+ l +".png"
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

h1_pe = 50

num_epochs = 10

#num_pe = [10, 20, 50, 75, 100]

#learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]

rate = 0.005

#act_func = ['relu', 'elu', 'selu', 'tanh', 'sigmoid', 'hard_sigmoid' ]

#opt = ['sgd', 'RMSprop', 'Adam', 'Adamax', 'Nadam', 'Adadelta']

sample_jump = [1,2,3,4,5]

col_list = []

for i in sample_jump:
    cols = []
    for j in range(784):
        if j % i == 0:
            cols.append(j)
    col_list.append(cols)
    

epochs = 10
act = 'relu'

mse_list = []
acc_list = []
time_list = []

#pca_list = [0.98, 0.95, 0.90, 0.80, 0.75]

len_vec = sample_jump

loss_plots = np.zeros((epochs, (len(len_vec))))

inp_num = []

print("Starting experiments")

for i, l in enumerate(col_list):
    
    x_train = x_full[0:sep]
    y_train = y_full[0:sep]
    
    x_train = x_train[:,l]
    
    x_test = x_full[sep:]
    
    x_test = x_test[:,l]
    y_test = y_full[sep:]
    
    scaler.fit(x_train)
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    num_inp = len(l)
    
    print("running for " + str(sample_jump[i]) + "sample jumps")
    t1 = time.time()
    
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=num_inp, activation='relu'))
    #model.add(Dense(l, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer= sgd, metrics=['mse','accuracy'])
    
    # Fit the model
    model_info = model.fit(x_train, y_train, epochs=epochs, batch_size=1)
    
    plt.figure()
    plot_model_history(model_info, str(sample_jump[i]))
    
    loss_plots[:,i] = model_info.history['loss']
    
    training_scores = model.evaluate(x_test, y_test)
    
    print("running for " + str(sample_jump[i]) + "sample jumps")
    print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2]*100))
    
    t2 = time.time()
    
    mse_list.append(training_scores[1]*100)
    acc_list.append(training_scores[2]*100)
    
    time_dur = t2 - t1
    
    time_list.append(time_dur)
    
    inp_num.append(num_inp)
    
#h1_arr = np.array(num_pe).reshape(len(num_pe),1)

h1 = [50] * len(len_vec)
h1_arr = np.array(h1).reshape(len(len_vec),1)

inp_num_arr = np.array(inp_num).reshape(len(len_vec),1)

sample_arr = np.array(sample_jump).reshape(len(len_vec),1)


lr = [0.005] * len(len_vec)
lr_arr = np.array(lr).reshape(len(len_vec),1)

act_func = [act] * len(len_vec)
act_arr = np.array(act_func).reshape(len(len_vec),1)

opt = ['SGD'] * len(len_vec)
opt_arr = np.array(opt).reshape(len(len_vec),1)

epoc = [10] * len(len_vec)
epoch_arr = np.array(epoc).reshape(len(len_vec),1)

mse_arr = np.array(mse_list).reshape(len(len_vec),1)
acc_arr = np.array(acc_list).reshape(len(len_vec),1)
time_arr = np.array(time_list).reshape(len(len_vec),1)

results = np.concatenate((h1_arr, act_arr, opt_arr, lr_arr, epoch_arr, inp_num_arr, sample_arr, mse_arr, acc_arr, time_arr), axis=1)    

result = pd.DataFrame(results)
result.to_csv("DS_exp.csv")

error_plots = pd.DataFrame(loss_plots)
error_plots.to_csv("error_plots_DS.csv")



