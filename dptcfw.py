import math
import json
import numpy as np
from numpy import random
#import dealdate
# from numpy import linalg as la
from scipy import linalg as la
from scipy.fftpack import fft,ifft
#import federated
#import function
from scipy.fftpack import fft, ifft
import numpy as np
import argparse
from numpy import linalg as la
from sklearn.metrics import mean_squared_error

def svd_econ(A):
    m, n = A.shape
    U, S, V = la.svd(A)
    V = V.T
    if m < n:
        V = V[:, 0:m]
    else:
        U = U[:,0:n]
    return U, S, V


def prox_tnn(Y, rho):
    tnn=0
    trank=0
    #rho=int(rho)
    n3, n1, n2 = Y.shape
    X = np.zeros((n3, n1, n2),dtype=np.complex128)
    Y = fft(Y,axis=0)
    tnn = 0
    trank = -1
   # print 'y',Y[0,:,:].shape
    U, S, V = svd_econ(Y[0,:, :])
   # print 'u',U.shape
   # print 's',S.shape
   # print 'v', V.shape
    r = -1
    for i in range(min(n1, n2)):
        if S[i] > rho:
            r = i
    if r >= 0:
        S = S[0:r + 1] - rho

    # first frontal slice
        X[0,:, :] = np.dot(np.dot(U[:, 0:r + 1], np.diag(S)), V[:, 0:r + 1].T)
        tnn = tnn + S.sum()
        trank = max(trank, r)

    if n3 % 2 == 0:
        halfn3 = int(n3 / 2)
    else:
        halfn3 = int((n3 + 1) / 2)
    for i in range(1, halfn3):
        U, S, V = svd_econ(Y[i,:, :])
        r = -1
        for j in range(min(n1, n2)):
            if S[j] > rho:
                r = j
        if r >= 0:
            S = S[0:r + 1] - rho
            X[i,:, :] = np.dot(np.dot(U[:, 0:r + 1], np.diag(S)), V[:, 0:r + 1].T)
            tnn = tnn + S.sum() * 2
            trank = max(trank, r)
        X[n3 - i,:, :] = np.conj(X[i,:, :])

    if n3 % 2 == 0:
        U, S, V = svd_econ(Y[halfn3,:, :])
        r = -1
        for ii in range(min(n1, n2)):
            if S[ii] > rho:
                r = ii
        if r >= 0:
            S = S[0:r + 1] - rho
            X[ halfn3,:, :] = np.dot(np.dot(U[:, 0:r + 1], np.diag(S)), V[:, 0:r + 1].T)
            tnn = tnn + S.sum()
            trank = max(trank, r)

    tnn = tnn / n3
    X = ifft(X,axis=0)
   # print tnn
    return X, tnn, trank + 1

X_List = []
V_List = []
F_List = []
c_list = []
Omega_list = []
Omega_list_tensor = []
data_predict = None
PredictWithParameter = []



def t_prod(A,B):
    [a3,a1,a2] = A.shape
    [b3,b1,b2] = B.shape
    A = fft(A,axis=0)
    B = fft(B,axis=0)
    C = np.zeros((b3,a1,b2))
    for i in range(b3):
        C[i,:,:] = np.dot(A[i,:,:],B[i,:,:])
    C = ifft(C,axis=0)
    return C

def SoftShrink( X, tau):
    z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

    return z

def SVDShrink(X, tau):
        W_bar = np.empty((X.shape[0], X.shape[1], 0), )
        m,n,k=np.shape(X)
        D = np.fft.fft(X,axis=0)
        for i in range(k):
            if i < k:
                U, S, V = np.linalg.svd(D[:, :, i], full_matrices=False)
                S = SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis=2)
            if i == k:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar,axis=0).real

def F_global(X,k,sigma,L,T,J):
    (k,m,n) = np.shape(X)
    ListInit(T,k,m,n)
    C=T/4
    XX = np.zeros((k, m, n),dtype=np.complex128)
    XY = np.zeros((k, m, n),dtype=np.complex128)
    A = np.zeros((k, m, n),dtype=np.complex128)
    Xomega=np.zeros(X.shape)
    lam=1000
    #lam=300-500 if FTC
    for iii in range(k):
        Xomega[iii,:,:]=Pomega(X[iii,:,:])
    for t in range(T+1):
        temp_v = random.uniform(0, 1)
        while temp_v == 0:
            temp_v = random.uniform(0, 1)
        for i in range(1,k+1):
            Xt_add1=XX[i-1,:,:]
            #print Xt_add1
            A[i-1,:,:]= F_local(i,Xt_add1,t,T,L,Pomega(X[i-1,:,:]))
        if t <T:
           # lamdaaaa=5*sum(A.shape)/2
           # print lamdaaaa
            E = random.normal(loc=0, scale=sigma, size=(k,m, n))
            A=A+E
            XX,TNN,trank=prox_tnn(A,lam)

       
def ListInit(T,k,m,n):
    global X_List,V_List,F_List,c_list,data_predict
    # X_List.clear()
    # V_List.clear()
    # F_List.clear()
    # c_list.clear()
    X_List=[]
    V_List=[]
    F_List=[]
    c_list=[]
    for i in range(k):
        X_List.append([np.zeros((m,n))]*(T+2))
        V_List.append([np.zeros((m,n))]*(T+2))
        F_List.append([np.zeros((m,n))]*(T+2))
        c_list.append(1)
    data_predict = np.zeros((k,m, n))
    for i in range(1,k+1):
        X0 = np.zeros((m, n))
        X1 = np.zeros((m, n))
        X_List[i-1][0] = X0
        X_List[i-1][1] = X1

        V0 = np.zeros((n, n))
        V1 = np.zeros((n, n))
        V_List[i-1][0] = V0
        V_List[i-1][1] = V1

def linsearch(X,D,S,T):
    a=np.power(np.linalg.norm((X-S),ord=2),2)
    b=2*np.sum(((X-D))*((S-X)))
   # print 'b',b
   # print 'a',a
    gamma=1.0/(T+1)
    if a!=0:
        if -b/(2*a)<0 or -b/(2*a)==0:
            gamma=1.0/(T+1)
        elif -b/(2*a)>1:
            gamma=1.0/(T+1)
        else:
            gamma=-b/(2*a)
    #print 'gamma',gamma
    return gamma


def F_local(i,Xt_add1,t,T,L,X_observed):
    global X_List,V_List,F_List,c_list,data_predict
    (m,n) = np.shape(X_observed)
    temp_v = random.uniform(0,1)
    while temp_v == 0:
        temp_v = random.uniform(0, 1)

    X_List[i - 1][t] = Xt_add1
 
    Xt=X_List[i-1][t]
    if t == 0:
        Xt_sub1 = np.zeros((m, n))
        Xt_sub1_omega=np.zeros((m, n))
    else:
        Xt_sub1=X_List[i-1][t-1]
    mu = linsearch(Xt_sub1, X_observed, Xt_add1,T)
    #mu=0.8
 
    Z = (1-mu)*Xt_sub1+mu * Xt_add1
    ZUG=Z-(Pomega(Z)-  X_observed)
 

    F_Xt = (la.norm(Pomega(X_List[i-1][t]-X_observed),ord=2)/2)
    F_List[i-1][t] = F_Xt

    #if t > 1:
        # if F_List[i-1][t] > F_List[i-1][t-1]:
        #     c_list[i-1] = 1
        # else:
        #     c_list[i-1] = c+1
    if t == T-1:
        data_predict[i-1,:,:] = Xt_add1
    
        return np.zeros((m, n))
    else:
        # return Y-np.dot(UG_tran,X_observed)+E,St_add1
        return ZUG



def t_svd(M):
    [n3,n1, n2] = M.shape
    D = np.zeros((n3,n1, n2))
    D = fft(M,axis=0)
    Uf = np.zeros((n3,n1, n1))
    Thetaf = np.zeros((n3,n1, n2))
    Vf = np.zeros((n3,n2, n2))
    tnn=0
    for i in range(n3):
        temp_U, temp_Theta, temp_V = la.svd(D[i,:, :], full_matrices=True)
        Uf[i,:, :] = temp_U
        Thetaf[i,:n2, :n2,] = np.diag(temp_Theta)
        tnn=tnn+Thetaf[i,:n2, :n2]
        Vf[i,:, :] = temp_V
    U = np.zeros((n3,n1, n1))
    Theta = np.zeros((n3,n1, n2))
    V = np.zeros((n3,n2, n2))
    U = ifft(Uf,axis=0).real
    Theta = ifft(Thetaf,axis=0).real
    V = ifft(Vf,axis=0).real
  #  print tnn
    return U, Theta, V,tnn

def t_svd_me(M):
    [k, m, n] = M.shape
    D = np.zeros((k, m, n))
    D = fft(M,axis=0)
    Uf = np.zeros((k, m, m))
    Thetaf = np.zeros((k, m, n))
    Vf = np.zeros((k, n, n))
    for i in range(k):
        temp_U, temp_Theta, temp_V = la.svd(D[i, :, :])
        Uf[i, :, :] = temp_U
        Thetaf[i, :n, :n] = np.diag(temp_Theta)
        Vf[i, :, :] = temp_V
    U = ifft(Uf,axis=0).real
    Theta = ifft(Thetaf,axis=0).real
    V = ifft(Vf,axis=0).real
    return U, Theta, V



#
def Judge_epsilon_delta(e,d):
    if e <= 2*math.log((1/d),10):
        return 1
    else:
        return 0



def BoundPomege(X):
    L_bound = 0
    (k,m,n) = np.shape(X)
    for i in range(k):
        #X_P = Pomega(X[i,:,:])
        X_P = X[i, :, :]
        L = la.norm(X_P,ord=2)
        L_bound = max(L,L_bound)
    return L_bound

def dataprocess(X):
    (k, m, n) = np.shape(X)
    for j in range(k):
        for i in range(m):
            mean=sum(X[j,i,:])/len(X[j,i,:])
            X[j,i,:]=X[j,i,:]-mean
    return X








def Pomega(matrix):
    global Omega_list
    m,n = np.shape(matrix)
    X_P = np.zeros((m,n))
    for index in Omega_list:
        i,j = index
        X_P[i,j] = matrix[i,j]
    return X_P

def Pomegaentire(tensor):
    global Omega_list
    (m,n) = np.shape(matrix)
    X_P = np.zeros((m,n))
    for index in Omega_list:
        i,j = index
        X_P[i,j] = matrix[i,j]
    return X_P

def rmse(target,prediction):
    error = []
    (k,m,n)=np.shape(target)
    for i in range(k):
        for j in range(m):
            for jj in range(n):
                error.append(target[i][j][jj] - prediction[i][j][jj])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    rm=np.sqrt(sum(squaredError)/(m*n*k))
    return rm

def GetGlobalOmega(m,n,per):
    global Omega_list
    Omega_list = []
    num = int(m*n*per)
    for tt in range(num):
        i = random.randint(0,m)
        j = random.randint(0,n)
        while [i,j] in Omega_list:
            i = random.randint(0,m)
            j = random.randint(0,n)
        Omega_list.append([i,j])

def rmsenew(target,predict):
    (k,m,n)=target.shape
    error=[0]*k
    total=[0]*k
    for i in range(k):
        #  print data_predict[i,:,:]
        error[i] = np.power(la.norm(target[i, :, :] - predict[i, :, :]), 2)
        total[i] = np.power(la.norm(target[i, :, :]), 2)
    error_T = sum(error) / sum(total)
    return error_T

def JsonFileSave(Data,filename):
    path = 'C:/Users/ChenKx/Desktop/federated/' + filename
    file = open(path, "w+")
    file.writelines(Data)
    file.close()

def JsonFileLoad(filename):
    DataLoad = []
    path = 'C:/Users/ChenKx/Desktop/federated/' + filename
    file = open(path, "r")
    lines = file.readlines()
    for line in lines:
        DataLoad.append(json.loads(line))
    file.close()
    return DataLoad

def DealFile(filename):
    path = 'C:/Users/ChenKx/Desktop/' + filename
    file = open(path, "r")
    file_w = open('C:/Users/ChenKx/Desktop/Federate_learn_predict/matrix_graph_adjacent_unique_0.json',"w")
    line = file.readline()
    Dataload = line.split('}{')
    for i in range(len(Dataload)):
        if i == 0:
            Dataload[i] = Dataload[i] + '}\n'
        elif i == len(Dataload)-1:
            Dataload[i] = '{' + Dataload[i] + '\n'
        else:
            Dataload[i] = '{' + Dataload[i] + '}\n'
    file_w.writelines(Dataload)
    file.close()
    file_w.close()

if __name__ == "__main__":
    # JsonFileLoad('tensor_graph_adjacent_unique_1.json')
    # exit(0)
    DataSave = []
    IterData = {'Para':[],'Error':0.0}
    #epsilon = [0.1,0.2,0.5,1,2,5,6]
    epsilon = [0.1]
    delta = pow(10,-6)
    #omega_percent = [n for n in np.arange(0.1, 1.1, 0.1)]
    omega_percent=[0.5]
    T_list = [n for n in range(0,1000,10)]
    #T_list=[200]
    m = 200
    n =200
    k = 20
    r = 1

    lam=max(max(m,n),k)/min(min(m,n),k)

    mini_error = 100000
    missing_rate = 0.2
    sparsity = 0.3


    error = [0]*(k)
    total=[0]*(k)
    graph= t_prod(np.random.rand(k, m, r), np.random.rand(k, r, n))
    maxco = 0
    for iii in range(k):
        gu, gs, gv = np.linalg.svd(graph[iii, :, :])
        #if maxco < max(gs):
        maxco = maxco+max(gs)
   # print 'mc', np.sqrt(float(maxco)/k)

    for e in epsilon:
        for per in omega_percent:
            GetGlobalOmega(m, n, per)
            L = BoundPomege(graph)
            if Judge_epsilon_delta(e,delta):
                flag = 0
                dp_para = [e, delta]
                for T in T_list:
                    sigma = np.sqrt((4 *  math.sqrt(2  * math.log((1 / dp_para[1]), 10))) /(dp_para[0]))
                    IterData['Para'] = [e,delta,per,T]
                    F_global(graph,k,sigma,L,T,lam)
                    error_T=rmsenew(graph,data_predict)
                    #print(graph)
                    IterData['Error'] = error_T
                    PredictWithParameter.append(IterData)
                    print(IterData)

    JsonFileSave(DataSave,'tensor_without_noise.json')

    print(PredictWithParameter)
