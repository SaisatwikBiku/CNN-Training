 # Neural Network implementation using only numpy

from __future__ import print_function

import numpy as np
import pickle

# ========== Net Layers, Activation, Loss - Begin ==========

# Fully connected layer
class FC ():
  def __init__ (self, D_in, D_out):
    self.cache = None
    #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
    self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
    self.b = {'val': np.random.randn(D_out), 'grad': 0}

  def _forward (self, X):
    out = np.dot(X, self.W['val']) + self.b['val']
    self.cache = X
    return out

  def _backward (self, dout):
    X = self.cache
    dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
    self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
    self.b['grad'] = np.sum(dout, axis=0)
    #self._update_params()
    return dX

  def _update_params (self, learning_rate=0.001):
    # Update the parameters
    self.W['val'] -= learning_rate * self.W['grad']
    self.b['val'] -= learning_rate * self.b['grad']

# Dropout layer
class Dropout():
  def __init__(self, p=1):
    self.cache = None
    self.p = p

  def _forward(self, X):
    M = (np.random.rand(*X.shape) < self.p) / self.p
    self.cache = X, M
    return X*M

  def _backward(self, dout):
    X, M = self.cache
    dX = dout*M/self.p
    return dX


# ========== Net Layers, Activation, Poolong, Loss - End ==========
