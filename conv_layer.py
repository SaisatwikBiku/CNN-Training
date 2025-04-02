
from __future__ import print_function

import numpy as np
import pickle

import im2col

# Conv layer
class Conv():
  def __init__(self, Cin, Cout, F, stride=1, padding=1, bias=True):
    self.Cin = Cin # number of input channels
    self.Cout = Cout # number of output channels
    self.F = F #filter size (e.g. 3 for 3x3 conv)
    self.S = stride
    #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
    # Xavier Initialization
    # http://philipperemy.github.io/xavier-initialization/
    self.W = {'val': np.random.normal(0.0, np.sqrt(2/Cin), (Cout,Cin,F,F)), 'grad': 0} 
    self.b = {'val': np.random.randn(Cout), 'grad': 0}
    self.cache = None
    self.pad = padding

  # https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
  def _forward(self, X):
    self.X = X
    N = X.shape[0]    
    W = self.W['val']
    b = self.b['val']
    n_filter, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * self.pad) / self.S + 1
    h_out = int (h_out)
    w_out = (w_x - w_filter + 2 * self.pad) / self.S + 1
    w_out = int (w_out)
    
    # Let this be 3x3 convolution with stride = 1 and padding = 1
    # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
    self.X_col = im2col.im2col_indices (X, h_filter, w_filter, padding=1, stride=1)
    # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
    W_col = W.reshape (n_filter, -1)

    # 20x9 x 9x500 = 20x500
    # b should have size 6 * 50176, 50176=64*1*28*28
    bb = np.tile(b, (self.X_col.shape[1], 1))
    bb = np.transpose (bb)
    out = W_col @ self.X_col + bb

    # Reshape back from 20x500 to 5x20x10x10
    # i.e. for each of our 5 images, we have 20 results with size of 10x10
    out = out.reshape (n_filter, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    self.cache = X
    return out
    
  '''  
  # N is batch_size
  # Cin is the input image channels = 1
  def _forward(self, X):
    X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
    (N, Cin, H, W) = X.shape
    H_ = H - self.F + 1
    W_ = W - self.F + 1
    Y = np.zeros((N, self.Cout, H_, W_))

    for n in range(N):
      for c in range(self.Cout):
        for h in range(H_):
          for w in range(W_):
            Y[n, c, h, w] = np.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]

    self.cache = X
    return Y
  '''  

  def _backward (self, dout):
    # dout (N,Cout,H_,W_)
    # W (Cout, Cin, F, F)
    
    W = self.W['val']
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ self.X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = im2col.col2im_indices (dX_col, self.X.shape, h_filter, w_filter, padding=1, stride=1)

    return dX #, dW, db
    
  '''
  def _backward(self, dout):
    # dout (N,Cout,H_,W_)
    # W (Cout, Cin, F, F)
    X = self.cache
    (N, Cin, H, W) = X.shape
    H_ = H - self.F + 1
    W_ = W - self.F + 1
    W_rot = np.rot90(np.rot90(self.W['val']))

    dX = np.zeros(X.shape)
    dW = np.zeros(self.W['val'].shape)
    db = np.zeros(self.b['val'].shape)

    # dW
    for co in range(self.Cout):
      for ci in range(Cin):
        for h in range(self.F):
          for w in range(self.F):
            dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

    # db
    for co in range(self.Cout):
      db[co] = np.sum(dout[:,co,:,:])

    dout_pad = np.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
    #print('dout_pad.shape: ' + str(dout_pad.shape))
    # dX
    for n in range(N):
      for ci in range(Cin):
        for h in range(H):
          for w in range(W):
            #print('self.F.shape: %s', self.F)
            #print('%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s' % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
            dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

    return dX
    '''
    