import numpy as np
from  numpy import random
random.seed(10)
import matplotlib.pyplot as plt
import pandas as pd
import math
def sig(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))

dt = pd.read_csv('diabetes2.csv',header=None)
dt = dt.iloc[1:, :]
dt = np.c_[np.ones((dt.shape[0], 1)), dt]   ## adding column of 1 for basis
random.shuffle(dt)  ## shuffing data
dt = dt.astype(float)
train , test = dt[:616,:] , dt[616:,:]  ## spliting data into 80% training data and 20% test data
train_x , train_y = train[:,:-1] , train[:,-1]
test_x , test_y = test[:,:-1] , test[:,-1]
def net_input(theta, x):
    # Compute  weighted sum of inputs
    return np.dot(x, theta)

def sig(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))

def prob(theta, x):
    # Returns  probability after passing through sigmoid
    return sig(net_input(theta, x))

def cost_function( theta, x, y):
    # Computes the cost function 
    m = float(x.shape[0])
    total_cost = -(1. / m) * np.sum(y * np.log(prob(theta, x)) + (1. - y) * np.log(1. - prob(theta, x)))
    return total_cost

def gradient( theta, x, y):
    # Computes the gradient 
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sig(net_input(theta,   x)) - y)

i=0 
lr= 0.000329999999999985
theta=np.array ([-5.399851014105276,0.12641708798933426,0.02775789776049815,-0.015348784659945427,0.00015536262044078656,-0.000455167240577413,0.047753967941772855,0.9269022952996796,-0.00038280664833698725])
cf=np.array([cost_function(theta,train_x,train_y)])  # cf is array of cost function
ia=np.array([0])
# print(gradient(theta,train_x,train_y))
a=math.sqrt(net_input(gradient(theta,train_x,train_y),gradient(theta,train_x,train_y)))
while i<100000:
    # print("cost function = ",cost_function(theta,train_x,train_y))
    if i%1000==0:
        print("itertion = ",i , end=' , ')
        print(math.sqrt(net_input(gradient(theta,train_x,train_y),gradient(theta,train_x,train_y))))
        print("cost function = ",cost_function(theta,train_x,train_y) ,end=' , ' )
        lr+=0.00001
        print(lr)
   
    
    
       
    temp=np.subtract(theta,(lr*gradient(theta,train_x,train_y)))
    theta=temp[:]
    i=i+1
    ia=np.append(ia,i)
    cf=np.append(cf,[cost_function(theta,train_x,train_y)])
    if math.sqrt(net_input(gradient(theta,train_x,train_y),gradient(theta,train_x,train_y))) > a:
        print("overshoot  ", lr,'iterion',i )
        lr-=0.000001


    a=math.sqrt(net_input(gradient(theta,train_x,train_y),gradient(theta,train_x,train_y)))
    
plt.plot(ia, cf)
plt.show()
for x in theta:
    print(x, end=',')
def predict(x):
    final_theta = theta[:]
    return prob(final_theta, x)

def accuracy(n,x,y):
    i=0
    a=1.
    b=0.
    count =0
    while i<152:
        if predict(x[i]) >= 0.5 and y[i] == a :
            count=count+1
        if predict(x[i]) < 0.5 and y[i] == b :
            count=count+1
        i=i+1
    return (count/n)*100

print("cost function = ",cost_function(theta,train_x,train_y))
print("correctness = ", accuracy(152,test_x,test_y)/100)
print("accuracy in test data = " , int((152*accuracy(152,test_x,test_y)/100)),"/152 (",accuracy(152,test_x,test_y),"%)")










