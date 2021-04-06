import numpy as np
from scipy.fftpack import fft,ifft
def t_prod_me(A,B):
    [a1,a2,a3] = A.shape
    [b1,b2,b3] = B.shape
    A = fft(A)
    B = fft(B)
    C = np.zeros((a1,b2,b3), dtype = complex)
    for i in range(b3):
        C[:, :,i] = np.dot(A[:, :, i], B[:, :, i])
    C = ifft(C)
    return C

def t_prod(A,B):
    [a3,a1,a2] = A.shape
    [b3,b1,b2] = B.shape
    A = fft(A,axis=0)
    B = fft(B,axis=0)
    C = np.zeros((b3,a1,b2))
    C[0,:,:]=np.dot(A[0,:,:],B[0,:,:])
    halfn3=np.round(b3/2)
    for i in range(1,halfn3):
        C[i,:,:] = np.dot(A[i,:,:],B[i,:,:])
        C[b3-i,:,:]=np.conj(C[i,:,:])
    if b3/2==0:
        i=halfn3+1
        C[i,:,:] = np.dot(A[i,:,:],B[i,:,:])
    C = ifft(C,axis=0)
    return C

def tran(X):
    k,m,n=X.shape
    Xtran=np.zeros((k,n,m))
    Xtran[0,:,:]=np.transpose(X[0,:,:])
    for i in range(1,k):
        Xtran[i,:,:]=np.transpose(X[k-i,:,:])
    return Xtran

def SVT(X,lam):
    [U, S, V] = svd_econ(X)
    (m) = np.shape(S)
    for i in range(int(m[0])):
        if S[i] > lam:
            S[i]=S[i] - lam
        else:
            S[i]=0
    New=np.dot(np.dot(U,np.diag(S)),np.transpose(V))
    return New

def svd_econ(A):
    m, n = A.shape
    U, S, V = np.linalg.svd(A)
    V = V.T
    if m < n:
        V = V[:, 0:m]
    else:
        U = U[:,0:n]
    return U, S, V