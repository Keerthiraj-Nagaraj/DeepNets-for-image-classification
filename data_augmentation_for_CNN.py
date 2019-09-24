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

## Keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.losses import categorical_crossentropy

from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator

def plot_model_history(model_history):
    plt.plot(range(1,len(model_history.history['loss'])+1), model_history.history['loss'])
    plt.title('Learning Curve')
    plt.xlabel('Iter #')
    plt.ylabel('Function value')
    plt.savefig("learning_curve_just_data_aug.png")
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
    plt.savefig('confusion_matrix_just_data_aug.png')


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


input_sha = (28, 28, 1)
num_classes = 10


x_full_reshaped = x_full.reshape(len(x_full), 28,28,1)

datagen = ImageDataGenerator()

transform_parameters = {'theta':np.random.randint(-30,30) , 
                        'tx':0.1 * np.random.rand() , 'ty':0.1* np.random.rand(), 
                        'zx': 0.9 + 0.3 * np.random.rand(), 'zy':0.9 + 0.3 * np.random.rand()}

#transform_parameters_2 = {'theta':10, 'tx':0.1, 'ty':0.1, 'zx':1.1, 'zy':0.9}

x_aug = np.zeros((3*len(x_full), 28, 28, 1))
y_aug = np.zeros((3*len(y_full),10))

cur_count = 0

for i in range(3*len(x_full)):
    
    if i%3 ==0:
        x_aug[i] = x_full_reshaped[cur_count]
        y_aug[i] = y_full[cur_count]
        #cur_count += 1
    elif i%3 == 1:
        temp = datagen.apply_transform(x_full_reshaped[cur_count], transform_parameters)
        x_aug[i] = temp
        y_aug[i] = y_full[cur_count]    
        
        transform_parameters = {'theta':np.random.randint(-30,30) , 
                        'tx':0.1 * np.random.rand() , 'ty':0.1* np.random.rand(), 
                        'zx': 0.9 + 0.3 * np.random.rand(), 'zy':0.9 + 0.3 * np.random.rand()}
    else:
        temp = datagen.apply_transform(x_full_reshaped[cur_count], transform_parameters)
        x_aug[i] = temp
        y_aug[i] = y_full[cur_count]    
        
        transform_parameters = {'theta':np.random.randint(-30,30) , 
                        'tx':0.1 * np.random.rand() , 'ty':0.1* np.random.rand(), 
                        'zx': 0.9 + 0.3 * np.random.rand(), 'zy':0.9 + 0.3 * np.random.rand()}

        cur_count += 1
        
imshow(x_aug[0].reshape(28,28))

imshow(x_aug[1].reshape(28,28))

imshow(x_aug[2].reshape(28,28))


x_aug_reshaped = x_aug.reshape(len(x_aug), 784)

X_aug =np.concatenate((x_aug_reshaped, y_aug), axis=1)

#scipy.io.savemat('Aug_data.mat', X_aug)

np.random.shuffle(X_aug)
sep=round(len(x_aug)*4/5)

x_full=X_aug[:,0:784]
y_full=X_aug[:,784:]

x_train = x_full[0:sep]
y_train = y_full[0:sep]

x_test = x_full[sep:]
y_test = y_full[sep:]

x_train_reshaped = x_train.reshape(len(x_train),28,28,1)
x_test_reshaped = x_test.reshape(len(x_test),28,28,1)


mse_list = []
acc_list = []
time_list = []

print("Starting experiments")

t1 = time.time()

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape= input_sha))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss= categorical_crossentropy,
              optimizer= optimizers.SGD(lr=0.1),
              metrics=['mse','accuracy']
              )


model_info = model.fit(x_train_reshaped, y_train, epochs=20, verbose=1, validation_split = 0.2)

plt.figure()
plot_model_history(model_info)

training_scores = model.evaluate(x_test_reshaped, y_test)

model.save('final_model_just_aug.h5')


print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2]*100))

t2 = time.time()

mse_list.append(training_scores[1]*100)
acc_list.append(training_scores[2]*100)

time_dur = t2 - t1

time_list.append(time_dur)

y_truth = y_test

y_pred = model.predict(x_test_reshaped)

y_tr = [ np.argmax(t) for t in y_truth ]
y_pr = [ np.argmax(t) for t in y_pred ]

cm = confusion_matrix(y_tr, y_pr)

target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(classification_report(y_tr, y_pr, target_names=target_names))


plot_confusion_matrix(cm, target_names, normalize=True, title='Confusion matrix')




