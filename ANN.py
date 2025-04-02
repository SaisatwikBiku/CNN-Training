
from __future__ import print_function

import numpy as np
import pickle

import nn_layer
import activation

# ========== ANN - Begin ==========

#Simple 2 layer NN
class TwoLayerNet ():
  def __init__(self, D_in, H, D_out, weights=''):
    '''
    D_in: input feature dimension
    H:    number of hidden neurons, output dimension of first FC layer and input dimension of second FC layer
    D_out: output dimension, which is 10 for digit recognition.
    '''
    self.FC1 = nn_layer.FC (D_in, H)
    self.ReLU1 = activation.ReLU()
    self.FC2 = nn_layer.FC (H, D_out)

    if weights == '':
      pass
    else:
      # Load weights from file
      with open (weights,'rb') as f:
        params = pickle.load(f)
        self.set_params(params)

  def forward(self, X):
    h1 = self.FC1._forward(X)
    a1 = self.ReLU1._forward(h1)
    h2 = self.FC2._forward(a1)
    return h2

  def backward(self, dout):
    dout = self.FC2._backward(dout)
    dout = self.ReLU1._backward(dout)
    dout = self.FC1._backward(dout)

  def get_params(self):
    return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

  def set_params(self, params):
    [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

#Simple 3 layer NN
#################################
########   TODO  ################
#################################
class ThreeLayerNet():
    def __init__(self, D_in, H1, H2, D_out, weights=''):
        self.FC1 = nn_layer.FC(D_in, H1)
        self.ReLU1 = activation.ReLU()
        self.FC2 = nn_layer.FC(H1, H2)
        self.ReLU2 = activation.ReLU()
        self.FC3 = nn_layer.FC(H2, D_out)
        if weights:
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1._forward(X)
        a1 = self.ReLU1._forward(h1)
        h2 = self.FC2._forward(a1)
        a2 = self.ReLU2._forward(h2)
        h3 = self.FC3._forward(a2)
        return h3

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.FC1._backward(dout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b = params
# ========== ANN - End ==========
