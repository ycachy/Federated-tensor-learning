import csv
import math
import scipy.io as scio
import numpy as np
import pandas as pd
import xlrd
import io
import matplotlib.pyplot as plt
import random

#Y=np.zeros((n_samples, n_features))
def Synthetic(n_sample,n_feature):
    u = np.random.random((n_sample,1))
    v = np.random.random((n_feature,1))
    YYY = np.dot(u,v.T )
    return YYY

def dataprocess(X):
    (m, n) = np.shape(X)
    for i in range(m):
        mean=sum(X[i,:])/len(X[i,:])
        X[i,:]=X[i,:]-mean
    return X

def Omegamatrix(Xx,samplenum):
    (m,n)=np.shape(Xx)
    X=np.zeros((m,n))
    for i in range(m):
        loc=np.random.randint(n,size=samplenum)
        for j in loc:
            X[i][j]=1
    Result=X*Xx
    return X,Result

def maxnorm(YY):
    (m,n)=np.shape(YY)
    maxx=0
    for i in range(m):
        temp=np.linalg.norm(YY[i,:],ord=2)
        if maxx<temp:
            maxx=temp
    return maxx

def update(Ome,n,k,v,lambda1,T,t,L,Yi,Di):
    YA = np.zeros((1,n))
    Ai = np.zeros(n)
    AN=np.zeros((n,n))
    if t==0:
        Yiii=np.zeros(n)
    else:
        Yiii=Yi
    Ai=(Yiii-Di)*Ome
    Aii=[]
    Aii.append(Ai)
    ui=np.dot(Aii,v)#/lambda1
    Yite=(1.0-1.0/T)*Yiii-(float(k)/T)*ui*np.transpose(v)
    if np.linalg.norm(Yite*Ome,ord=2)!=0:
        if L/np.linalg.norm(Yite*Ome,ord=2)<1 :
            YA=(L/np.linalg.norm(Yite*Ome,ord=2))*Yite
        else:
            YA=Yite
    Ai=(YA-Di)*Ome
    #print 'ai',Ai
    if t!=0:
        Aii=[]
        Aii.append(Ai)
        AN=np.dot(np.transpose(Aii),Aii)
    else:
        AN = np.dot(np.transpose(Ai), Ai)
    #print AN.shape
    return YA,AN


def rmse(target,prediction):
    error = []
    (m,n)=np.shape(target)
    for i in range(m):
        for j in range(n):
            error.append(target[i][j] - prediction[i][j])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    rm=np.sqrt(sum(squaredError) / (m*n))
    return rm


#------------------------------glocal-----
def glocalmc(datainput,eplison,TTT):
    YY=[]

    (m, n) = np.shape(datainput)
    for TT in TTT:
       # print TT
    #for eplisoni in eplison:
        v=np.zeros((n,1))
        lamda=0
        Y=np.zeros((m,n))
        #TT=int(TT*eplison)
        for t in range(TT):
            sigma = math.pow(L,2) * np.sqrt(64*TT* np.log10(1 / delta)) / (eplison*m)
            #sigma=0
            #print sigma
            W=np.zeros((n,n))
            lambda1=lamda+math.sqrt(sigma*np.log10(float(n)/belta))*math.pow(n,1.0/4)
            for i in range(m):
                Di=datainput[i,:]
                Xomegai=Xomega[i,:]
                Yii,AN=update(Xomegai,n,k,v,lambda1,TT,t,L,Y[i,:],Di)
                #print AN
                Y[i,:]=Yii
                W=W+AN
           # print 'w',W
            for it in range(n):
                for jt in range(n):
                    W[it][jt]=W[it][jt]+ random.gauss(0,sigma)
            lamdasqrt, vv = np.linalg.eig(W)
            lamda=np.sqrt(lamdasqrt[0])
            v=np.real(vv[:,0])

        YY.append(Y)
    return YY


if __name__=='__main__':
#-----synthetic
    n_sample, n_feature = 5000, 40
    datainput1=Synthetic(n_sample,n_feature)

    #------------------------


    #-------parameters
    delta=math.pow(10,-6)
    eplison=[0.1,0.5,1.0,2.0,5.0]
    TT=60
    #TT = [jj for jj in range(5,100,5)]
    belta = 10
    rate=0.3
    #------------------------

    samplenumber=rate*n_feature
    Xomega,datainput=Omegamatrix(datainput1,int(samplenumber))
    #print datainput1
    L =float(maxnorm(datainput))
    #print L
    du,ds,dv=np.linalg.svd(datainput1)
    k=sum(ds)
    #k=100

    di=[]
    YY=glocalmc(datainput,eplison,TT)
    #print YY
    for ii in range(len(YY)):
        di.append(rmse(YY[ii],datainput1))
    print di



