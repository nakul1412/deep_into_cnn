import numpy as np
from  numpy import random
random.seed(10)
import matplotlib.pyplot as plt
import pandas as pd
import math


dt = pd.read_csv('diabetes2.csv',header=None)
dt = dt.iloc[1:, :]
dt = np.c_[np.ones((dt.shape[0], 1)), dt]   ## adding column of 1 for basis
random.shuffle(dt)  ## shuffing data
dt = dt.astype(float)
train , test = dt[:616,:] , dt[616:620,:]  ## spliting data into 80% training data and 20% test data
train_x , train_y = train[:,:-1] , train[:,-1]
test_x , test_y = test[:,:-1] , test[:,-1]

def sig(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))

def sig2(x):
    # sigmoid function
    return (1 + np.exp(-x))

def sig1(x):
    # sigmoid function
    return  np.exp(-x)

def net_input(theta, x):
    # Compute  weighted sum of inputs
    return np.dot(x, theta)

theta=np.array ([1,2,3,4,5])
theta1=np.array ([1,2,3,4,5])
theta2=theta*theta
print(test_x)
print(test_x.T)
