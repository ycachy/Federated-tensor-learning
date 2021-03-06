import numpy as np
from scipy.fftpack import fft,ifft
def t_prod(A,B):
    [a1,a2,a3] = A.shape
    [b1,b2,b3] = B.shape
    A = fft(A)
    B = fft(B)
    C = np.zeros((a1,b2,b3), dtype = complex)
    for i in range(b3):
        C[:, :,i] = np.dot(A[:, :, i], B[:, :, i])
    C = ifft(C)
    return C
