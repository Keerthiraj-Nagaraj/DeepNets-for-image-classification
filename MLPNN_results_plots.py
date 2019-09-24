# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:10:55 2018

@author: keert
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Learning curves for 1st hidden layer topology

h1_topo = pd.read_csv("error_plots_h1_topo.csv")
h1_topo = np.array(h1_topo.iloc[:,1:6])

h1_topo_exp = pd.read_csv("1_HL_topo_exp.csv")
h1_topo_exp = np.array(h1_topo_exp.iloc[:,1:5])


leg = [str(i) for i in h1_topo_exp[:,0]]

plt.figure()

for i in range(h1_topo.shape[1]):
    plt.plot(h1_topo[:,i], label = 'err')
    plt.grid(True)
    plt.title('Learning curves for various number of PEs in 1st hidden layer')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("1_HL_topo_LC.png")
          
i = 0
               
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.plot(h1_topo_exp[:,i], h1_topo_exp[:,i+1])
plt.title("MSE Vs Number of PEs in 1st hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.plot(h1_topo_exp[:,i], h1_topo_exp[:,i+2])
plt.title("Accuracy (percent) Vs Number of PEs in 1st hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.plot(h1_topo_exp[:,i], h1_topo_exp[:,i+3])
plt.title("Training Duration Vs Number of PEs in 1st hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("1_HL_topo_results.png")

#################################################


# Learning curves for 2nd hidden layer topology

h2_topo = pd.read_csv("error_plots_h2_topo.csv")
h2_topo = np.array(h2_topo.iloc[:,1:6])

h2_topo_exp = pd.read_csv("2_HL_topo_exp.csv")
h2_topo_exp = np.array(h2_topo_exp.iloc[:,1:6])

i = 1

leg = [str(i) for i in h2_topo_exp[:,i]]

plt.figure()

for i in range(h2_topo.shape[1]):
    plt.plot(h2_topo[:,i], label = 'err')
    plt.grid(True)
    plt.title('Learning curves for various number of PEs in 2nd hidden layer')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("2_HL_topo_LC.png")

i = 1
               
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.plot(h2_topo_exp[:,i], h2_topo_exp[:,i+1])
plt.title("MSE Vs Number of PEs in 2nd hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.plot(h2_topo_exp[:,i], h2_topo_exp[:,i+2])
plt.title("Accuracy (percent) Vs Number of PEs in 2nd hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.plot(h2_topo_exp[:,i], h2_topo_exp[:,i+3])
plt.title("Training Duration Vs Number of PEs in 2nd hidden layer")
plt.xlabel("Number of PEs")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("2_HL_topo_results.png")




#################################################


# Results for step size

lr_sgd = pd.read_csv("error_plots_exp_lr_sgd.csv")
lr_sgd = np.array(lr_sgd.iloc[:,1:6])

lr_sgd_exp = pd.read_csv("lr_exp_sgd.csv")
lr_sgd_exp = np.array(lr_sgd_exp.iloc[:,4:8])


i = 0
leg = [str(i) for i in lr_sgd_exp[:,i]]

plt.figure()

for i in range(lr_sgd.shape[1]):
    plt.plot(lr_sgd[:,i], label = 'err')
    plt.grid(True)
    plt.title('Learning curves for different step sizes')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("lr_sgd_LC.png")

i = 0

my_xticks = leg

xnum = range(len(leg))
     
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+1])
plt.title("MSE Vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+2])
plt.title("Accuracy (percent) Vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+3])
plt.title("Training Duration Vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("lr_sgd_results.png")



#################################################


# Results for PCA

data = pd.read_csv("error_plots_pca.csv")
data = np.array(data.iloc[:,1:6])

data_exp = pd.read_csv("pca_exp.csv")
data_exp = np.array(data_exp.iloc[:, (data_exp.shape[1]-4) :data_exp.shape[1]])

i = 0

leg = [str(i) for i in data_exp[:,i]]

plt.figure()

for i in range(data.shape[1]):
    plt.plot(data[:,i], label = 'err')
    plt.grid(True)
    plt.title('Learning curves for PCA experiments')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("pca_LC.png")

i = 0

my_xticks = leg

xnum = range(len(leg))
     
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+1])
plt.title("MSE Vs % info retained")
plt.xlabel("% info retained")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+2])
plt.title("Accuracy (percent) Vs % info retained")
plt.xlabel("% info retained")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+3])
plt.title("Training Duration Vs % info retained")
plt.xlabel("% info retained")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("pca_results.png")


#################################################



# Results for Down Sampling

data = pd.read_csv("error_plots_DS.csv")
data = np.array(data.iloc[:,1:6])

data_exp = pd.read_csv("DS_exp.csv")
data_exp = np.array(data_exp.iloc[:, (data_exp.shape[1]-5) :data_exp.shape[1]])



leg = [str(i) for i in data_exp[:,i]]

plt.figure()

for i in range(data.shape[1]):
    plt.plot(data[:,i])
    plt.grid(True)
    plt.title('Learning curves for Down Sampling experiments')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("DS_LC.png")

i = 1

my_xticks = leg

xnum = range(len(leg))
     
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+1])
plt.title("MSE Vs # of input features")
plt.xlabel("# of input features")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+2])
plt.title("Accuracy (percent) Vs # of input features")
plt.xlabel("# of input features")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+3])
plt.title("Training Duration Vs # of input features")
plt.xlabel("# of input features")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("DS_results.png")



#################################################



# Results for Activation Function

data = pd.read_csv("error_plots_exp_act_func.csv")
data = np.array(data.iloc[:,1:7])

data_exp = pd.read_csv("act_func_exp.csv")

data_exp = np.array(data_exp.iloc[:, (data_exp.shape[1]-3) :data_exp.shape[1]])

leg_here = pd.read_csv("act_func_exp.csv")
leg_here = leg_here.iloc[:,3]

leg = list(leg_here.values)

plt.figure()

for i in range(data.shape[1]):
    plt.plot(data[:,i])
    plt.grid(True)
    plt.title('Learning curves for different activation functions')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("act_func_LC.png")

i = -1

my_xticks = leg

xnum = range(len(leg))

xname = 'Activation Function'

plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+1])
plt.title("MSE Vs " + xname)
plt.xlabel(xname)
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+2])
plt.title("Accuracy (percent) Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+3])
plt.title("Training Duration Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("act_func_results.png")



#################################################



# Results for Optimizer

data = pd.read_csv("error_plots_exp_opt.csv")
data = np.array(data.iloc[:,1:7])

data_exp = pd.read_csv("opt_exp.csv")

data_exp = np.array(data_exp.iloc[:, (data_exp.shape[1]-3) :data_exp.shape[1]])

leg_here = pd.read_csv("opt_exp.csv")
leg_here = leg_here.iloc[:,3]

leg = list(leg_here.values)

plt.figure()

for i in range(data.shape[1]):
    plt.plot(data[:,i])
    plt.grid(True)
    plt.title('Learning curves for different optimizers')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("opt_LC.png")

i = -1

my_xticks = leg

xnum = range(len(leg))

xname = 'Optimizer'

plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+1])
plt.title("MSE Vs " + xname)
plt.xlabel(xname)
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+2])
plt.title("Accuracy (percent) Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+3])
plt.title("Training Duration Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("opt_results.png")

#####################################################################

# Results for step size for ADAMAX optimizer

lr_sgd = pd.read_csv("error_plots_exp_lr.csv")
lr_sgd = np.array(lr_sgd.iloc[:,1:6])

lr_sgd_exp = pd.read_csv("lr_exp.csv")
lr_sgd_exp = np.array(lr_sgd_exp.iloc[:,4:8])


i = 0
leg = [str(i) for i in lr_sgd_exp[:,i]]

plt.figure()

for i in range(lr_sgd.shape[1]):
    plt.plot(lr_sgd[:,i], label = 'err')
    plt.grid(True)
    plt.title('Learning curves Vs step sizes - Adamax optimizer')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,9))
               
plt.legend(leg)
plt.savefig("lr_adamax_LC.png")

i = 0

my_xticks = leg

xnum = range(len(leg))
     
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+1])
plt.title("MSE Vs Step Size (ADAMAX)")
plt.xlabel("Step Size")
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+2])
plt.title("Accuracy (percent) Vs Step Size (ADAMAX)")
plt.xlabel("Step Size")
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(lr_sgd_exp[:,i+3])
plt.title("Training Duration Vs Step Size (ADAMAX)")
plt.xlabel("Step Size")
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("lr_adamax_results.png")


#################################################



# Results for epoch numbers

epoch_num = [10,20,30,40,50]

data = pd.read_csv("error_plots_exp_epochs.csv")
data = np.array(data.iloc[:,1:6])

data_exp = pd.read_csv("epochs_exp.csv")

data_exp = np.array(data_exp.iloc[:, (data_exp.shape[1]-4) :data_exp.shape[1]])

leg = [str(i) for i in epoch_num]


plt.figure()

for i in range(data.shape[1]):
    plt.plot(data[:epoch_num[i],i])
    plt.grid(True)
    plt.title('Learning curves for different optimizers')
    plt.xlabel("Epoch #")
    plt.ylabel("Function value")
    plt.xlim((0,49))
               
plt.legend(leg)
plt.savefig("epoch_LC.png")

i = 0

my_xticks = leg

xnum = range(len(leg))

xname = 'Epoch numbers'

plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+1])
plt.title("MSE Vs " + xname)
plt.xlabel(xname)
plt.ylabel("MSE (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,2)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+2])
plt.title("Accuracy (percent) Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Accuracy (percent)")
plt.grid(True)

#plt.figure()
plt.subplot(3,1,3)
plt.xticks(xnum, my_xticks)
plt.plot(data_exp[:,i+3])
plt.title("Training Duration Vs " + xname)
plt.xlabel(xname)
plt.ylabel("Training Duration (seconds)")
plt.grid(True)
plt.savefig("epoch_results.png")

#####################################################################