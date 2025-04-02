
from __future__ import print_function

import numpy as np


class MaxPool ():
  def __init__(self, F, stride):
    self.F = F
    self.S = stride
    self.cache = None

  def _forward (self, X):
    # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
    # N is batch_size
    # Cin is the input image channels = 1
    (N,Cin,H,W) = X.shape
    F = self.F
    W_ = int(float(W)/F)
    H_ = int(float(H)/F)
    #Y = np.zeros((N,Cin,W_,H_)) #Ning: error?
    Y = np.zeros((N,Cin,H_,W_))
    M = np.zeros(X.shape) # mask
    for n in range(N):
      for cin in range(Cin):
        for h_ in range(H_):
          for w_ in range(W_):
            Y[n,cin,h_,w_] = np.max(X[n,cin,F*h_:F*(h_+1),F*w_:F*(w_+1)])
            i,j = np.unravel_index(X[n,cin,F*h_:F*(h_+1),F*w_:F*(w_+1)].argmax(), (F,F))
            M[n,cin,F*h_+i,F*w_+j] = 1
    self.cache = M
    return Y

  def _backward (self, dout):
    M = self.cache
    (N,Cin,H,W) = M.shape
    dout = np.array(dout)
    #print('dout.shape: %s, M.shape: %s' % (dout.shape, M.shape))
    dX = np.zeros(M.shape)
    for n in range(N):
      for c in range(Cin):
        #print('(n,c): (%s,%s)' % (n,c))
        dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
    return dX*M
