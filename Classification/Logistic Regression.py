# Logistic Regression
import numpy as np

# Define Sigmoid Function
def sig(x):
    return 1./(1+np.exp(-x))

# Training function
def lr_train( feature, label, alpha, maxcycle):
    n=np.shape(feature)[1]
    w=np.mat(np.ones(n))
    i=0
    while i<=maxcycle:
        i+=1
        hypothesis=sig(feature*w)
        error=hypothesis-label
        if i%100==0:
            print("In "+str(i)+"th cycle,training error rate is "+str(error_rate(hypothesis,label)))
        w=w-alpha*error.T*feature
    return w

# rate of error
def error_rate(h,label):
    m=np.shape(h)[0]
    error_sum=0.0
    for i in range(m):
        assert  h[i,0]>0 and (1-h[i,0])>0
        error_sum-=(label[i,0]*np.log(h[i,0])+(1-label[i,0])*np.log(1-h[i,0]))
    return error_sum/m

# Main function
if __name__=="__main__":
    print("----------1.load data----------")
    feature,label=load_data("data.text") # Need further coding.
    print("----------2.training----------")
    w=lr_train(feature,label,0.01,1000)
    print("----------3.save model----------")
    save_model("weights",w)

def load_data(file_name):
    f=open(file_name)
    feature_data=[]
    label_data=[]
    for line in f.readlines():
        feature_tmp=[]
        label_tmp=[]
        lines=line.strip().split("\t")
        feature_tmp.append(1)
        for i in xrange(len(lines)-1):
            feature_tmp.append(float(liens[i]))
        label_tmp.append(float(liens[-1]))

        feature_data.append(feature_tmp)
        albel_data.append(label_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data)

def save_model(file_name,w):
    m=np.shape(w)[0]
    f_w=open(file_name,"w")
    w_array=[]
    for i in xrange(m):
        w_array.append(str(w[i,0]))
    f_w.write("\t".join(w_array))
    f_w.close()