# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:57:33 2018

@author: keert
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

import scipy.io
import time

## Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def plot_model_history(model_history):
    plt.plot(range(1,len(model_history.history['loss'])+1), model_history.history['loss'])
    plt.title('Learning Curve')
    plt.xlabel('Iter #')
    plt.ylabel('Function value')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_norm_grey.png')



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

h1_pe = 100

num_pe = [1]

mse_list = []
acc_list = []
time_list = []

print("Starting Topology experiments")

t1 = time.time()

# create model
model = Sequential()
model.add(Dense(100, input_dim=784, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


sgd = optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
    
model.compile(loss='mean_squared_error', optimizer= sgd, metrics=['mse','accuracy'])

model_info = model.fit(x_train, y_train, epochs=50, batch_size=1)

plt.figure()
plot_model_history(model_info)

training_scores = model.evaluate(x_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2]*100))

t2 = time.time()

mse_list.append(training_scores[1]*100)
acc_list.append(training_scores[2]*100)

time_dur = t2 - t1

time_list.append(time_dur)

y_truth = y_test

y_pred = model.predict(x_test)

y_tr = [ np.argmax(t) for t in y_truth ]
y_pr = [ np.argmax(t) for t in y_pred ]

cm = confusion_matrix(y_tr, y_pr)

target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(classification_report(y_tr, y_pr, target_names=target_names))


plot_confusion_matrix(cm, target_names, normalize=True, title='Confusion matrix')


#
#
##h1_arr = np.array(num_pe).reshape(len(num_pe),1)
#h1 = [h1_pe] * len(num_pe)
#
#h1_arr = np.array(h1).reshape(len(num_pe),1)
#h2_arr = np.array(num_pe).reshape(len(num_pe),1)
#
#mse_arr = np.array(mse_list).reshape(len(num_pe),1)
#acc_arr = np.array(acc_list).reshape(len(num_pe),1)
#time_arr = np.array(time_list).reshape(len(num_pe),1)
#
#results = np.concatenate((h1_arr, h2_arr, mse_arr, acc_arr, time_arr), axis=1)    
#
#result = pd.DataFrame(results)
##result.to_csv("2_HL_topo_exp.csv")
#




