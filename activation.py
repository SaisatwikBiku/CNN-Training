
from __future__ import print_function

import numpy as np


# ReLU activation layer
class ReLU():
  def __init__ (self):
    self.cache = None

  def _forward (self, X):
    out = np.maximum(0, X)
    self.cache = X
    return out

  def _backward (self, dout):
    X = self.cache
    #dX = np.array(dout, copy=True)
    dX = dout.copy()
    dX[X <= 0] = 0
    return dX

# Sigmoid activation layer
class Sigmoid():
  def __init__ (self):
    self.cache = None

  def _forward (self, X):
    self.cache = X
    return 1 / (1 + np.exp(-X))

  def _backward (self, dout):
    X = self.cache
    dX = dout*X*(1-X)
    return dX

# tanh activation layer
class tanh():
  def __init__ (self):
    self.cache = X

  def _forward (self, X):
    self.cache = X
    return np.tanh(X)

  def _backward (self, X):
    X = self.cache
    dX = dout*(1 - np.tanh(X)**2)
    return dX

# Softmax activation layer
class Softmax():
  def __init__ (self):
    self.cache = None

  def _forward (self, X):
    maxes = np.amax(X, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    Y = np.exp(X - maxes)
    Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
    self.cache = (X, Y, Z)
    return Z # distribution

  def _backward (self, dout):
    X, Y, Z = self.cache
    dZ = np.zeros(X.shape)
    dY = np.zeros(X.shape)
    dX = np.zeros(X.shape)
    N = X.shape[0]
    for n in range(N):
      i = np.argmax(Z[n])
      dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
      M = np.zeros((N,N))
      M[:,i] = 1
      dY[n,:] = np.eye(N) - M
    dX = np.dot(dout,dZ)
    dX = np.dot(dX,dY)
    return dX
