# Logistic Regression
import numpy as np

# Define Sigmoid Function
def sig(x):
    return 1./(1+np.exp(-x))

def lr_train_bgd(feature,label,maxCycle,alpha):
    n=np.shape(feature)[1] # the number of features
    w=np.mat(np.ones((n,1))) # Define weights
    i=0
    while i<= maxCycle:
        i+=1
        hypothesis=sig(np.dot(feature,w))
        error=label-hypothesis
        if i%100==0:
            print("\t---------iter="+str(i)+", train error rate= " + str(error_rate(hypothesis,label)))
        w=w+alpha*feature.T*error
    return w