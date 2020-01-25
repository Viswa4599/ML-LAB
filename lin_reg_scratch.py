import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

airfoil = pd.read_csv('airfoil_self_noise.dat',sep='\t',header=None)

labels = airfoil[ 5]
features = airfoil.drop([5],axis  =1)


#Normalising features and labels to same scale
def normalize(data):
   return (data - data.mean())/data.std()

#Applying train test split
x_train,x_test,y_train,y_test = tts(features,labels,test_size  = 0.33, random_state = 42)

n = len(features.columns)
x_train = normalize(x_train)
x_test = normalize(x_test)
x_train.to_numpy()
x_test.to_numpy()
y_train.to_numpy()
y_test.to_numpy()



x_train = np.hstack((np.ones((x_train.shape[0],1)),x_train))
x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))

#Multivariate cost function
def cost_function(loss,m):
    J = np.sum(loss ** 2)/(2 * m)
    return J

def batch_gradient_descent(X, Y, B, alpha, iterations):
    cost_history = {}
    m = len(Y)
    k = 0
    for iteration in range(iterations):
    #print(iteration)
    # Hypothesis Values
        h = X.dot(B)
    # Difference b/w Hypothesis and Actual Y
        loss = h -Y
    # Gradient Calculation
        gradient = X.T.dot(loss) / m
    # Changing Values of B using Gradient
        B = B -alpha * gradient
    # New Cost Value
        cost = cost_function(loss,m)
        cost_history.update({k:cost})
        k+=1
    return B, cost_history


B = np.zeros(x_train.shape[1])
alpha = 0.02
iter_ = 1000
newB, cost_history = batch_gradient_descent(x_train, y_train, B, alpha, iter_)

print(cost_history)

def pred(x_test,theta):
    y_pred = x_test.dot(theta)
    return y_pred


y_pred = pred(x_test,newB)

def rmse(y_,y):
    return np.sqrt(((y_-y)**2).mean())

def r2(y_,y):
    sst = np.sum((y-y.mean())**2)
    ssr = np.sum((y_-y)**2)
    r2 = 1-(ssr/sst)
    return(r2)
#-----------------

print("Cost at convergence: ",cost_history[iter_ -1])
print("R2 Score: ",r2(y_pred,y_test))
print("RMSE Score: ",rmse(y_pred,y_test))

import matplotlib.pyplot as plt

plt.plot(*zip(*sorted(cost_history.items())))
plt.show()