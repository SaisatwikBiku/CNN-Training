
from __future__ import print_function

import numpy as np
import pickle

import activation
import pooling
import nn_layer
import conv_layer

# ========== LeNet5 - Begin ==========

'''
https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb
LeNet-5 Input
The LeNet architecture accepts a 32x32xC image as input,
where C is the number of color channels.
Since MNIST images are grayscale, C is 1 in this case.
Need to zero-pad the 28x28 input image into 32x32

Architecture
Layer 1: Convolutional. The output shape should be 28x28x6.
  Activation. Your choice of activation function.
  Pooling. The output shape should be 14x14x6.
Layer 2: Convolutional. The output shape should be 10x10x16.
  Activation. Your choice of activation function.
  Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Layer 3: Fully Connected. This should have 120 outputs.
  Activation. Your choice of activation function.
Layer 4: Fully Connected. This should have 84 outputs.
  Activation. Your choice of activation function.
Layer 5: Fully Connected (Logits). This should have 10 outputs (10 classes for number digits).

# See https://wiseodd.github.io/techblog/2016/07/16/convnet-conv_layer-layer/
#   N is the number of input
#   C = 1 (number of image channel, =1 for grayscale image)
#   H is the height of image
#   W is the width of the image
#   NF is the number of filter in the filter map W
#   HF = 3 is the height of the filter
#   WF = 3 is the width of the filter.
# input X: N*C*H*W
# input filter W: NF*C*HF*HW
# bias b: F*1
# See https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
'''
class LeNet5 ():
  def __init__(self):
    self.Cin = 1
    self.D_out = 10
    # Cin: input channel
    # Cout: output channel
    # F: kernel size 3x3
    # Conv1: Cin=1, Cout=6, F=3
    self.conv1 = conv_layer.Conv (self.Cin, 6, 3)
    self.ReLU1 = activation.ReLU ()
    self.pool1 = pooling.MaxPool (2,2)
    # Conv2: Cin=6, Cout=16, F=3
    self.conv2 = conv_layer.Conv (6, 16, 3)
    self.ReLU2 = activation.ReLU ()
    self.pool2 = pooling.MaxPool (2,2)
    # FC1 flatten to be 64*784
    self.FC1 = nn_layer.FC(784, 120)
    self.ReLU3 = activation.ReLU ()
    self.FC2 = nn_layer.FC (120, 84)
    self.ReLU4 = activation.ReLU ()
    self.FC3 = nn_layer.FC (84, self.D_out)
    self.Softmax = activation.Softmax ()

    self.p2_shape = None


  def forward (self, X):
    h1 = self.conv1._forward(X)
    a1 = self.ReLU1._forward(h1)
    p1 = self.pool1._forward(a1)
    h2 = self.conv2._forward(p1)
    a2 = self.ReLU2._forward(h2)
    p2 = self.pool2._forward(a2)
    # p2 shape = batch 64 * 16 k * 7 * 7 (7*7 is after pooling)
    self.p2_shape = p2.shape
    fl = p2.reshape(p2.shape[0],-1) # Flatten
    # fl shape = batch 64 * 784 = 16*7*7
    h3 = self.FC1._forward(fl)
    a3 = self.ReLU3._forward(h3)
    h4 = self.FC2._forward(a3)
    a5 = self.ReLU4._forward(h4)
    h5 = self.FC3._forward(a5)
    a5 = self.Softmax._forward(h5)
    return a5

  def backward (self, dout):
    #dout = self.Softmax._backward(dout)
    dout = self.FC3._backward(dout)
    dout = self.ReLU4._backward(dout)
    dout = self.FC2._backward(dout)
    dout = self.ReLU3._backward(dout)
    dout = self.FC1._backward(dout)
    dout = dout.reshape(self.p2_shape) # reshape
    dout = self.pool2._backward(dout)
    dout = self.ReLU2._backward(dout)
    dout = self.conv2._backward(dout)
    dout = self.pool1._backward(dout)
    dout = self.ReLU1._backward(dout)
    dout = self.conv1._backward(dout)

  def get_params(self):
    return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

  def set_params(self, params):
    [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params

# ========== LeNet5 - End ==========



# ========== CNN - Begin ==========


import loss_fun
import layer as l


class NeuralNet(object):

    loss_funs = dict(
        cross_ent=loss_fun.cross_entropy,
        hinge=loss_fun.hinge_loss,
        squared=loss_fun.squared_loss,
        l2_regression=loss_fun.l2_regression,
        l1_regression=loss_fun.l1_regression
    )

    dloss_funs = dict(
        cross_ent=loss_fun.dcross_entropy,
        hinge=loss_fun.dhinge_loss,
        squared=loss_fun.dsquared_loss,
        l2_regression=loss_fun.dl2_regression,
        l1_regression=loss_fun.dl1_regression
    )

    forward_nonlins = dict(
        relu=l.relu_forward,
        lrelu=l.lrelu_forward,
        sigmoid=l.sigmoid_forward,
        tanh=l.tanh_forward
    )

    backward_nonlins = dict(
        relu=l.relu_backward,
        lrelu=l.lrelu_backward,
        sigmoid=l.sigmoid_backward,
        tanh=l.tanh_backward
    )

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        if loss not in NeuralNet.loss_funs.keys():
            raise Exception('Loss function must be in {}!'.format(NeuralNet.loss_funs.keys()))

        if nonlin not in NeuralNet.forward_nonlins.keys():
            raise Exception('Nonlinearity must be in {}!'.format(NeuralNet.forward_nonlins.keys()))

        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss
        self.forward_nonlin = NeuralNet.forward_nonlins[nonlin]
        self.backward_nonlin = NeuralNet.backward_nonlins[nonlin]
        self.mode = 'classification'

        if 'regression' in loss:
            self.mode = 'regression'

    def train_step(self, X_train, y_train):
        """
        Single training step over minibatch: forward, loss, backprop
        """
        y_pred, cache = self.forward(X_train, train=True)
        loss = self.loss_funs[self.loss](self.model, y_pred, y_train, self.lam)
        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def predict_proba(self, X):
        score, _ = self.forward(X, False)
        return loss_fun.softmax(score)

    def predict(self, X):
        if self.mode == 'classification':
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            score, _ = self.forward(X, False)
            y_pred = np.round(score)
            return y_pred

    def forward(self, X, train=False):
        raise NotImplementedError()

    def backward(self, y_pred, y_train, cache):
        raise NotImplementedError()

    def _init_model(self, D, C, H):
        raise NotImplementedError()


class ConvNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # Conv-1
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = l.relu_forward(h1)

        # Pool-1
        hpool, hpool_cache = l.maxpool_forward(h1)
        h2 = hpool.ravel().reshape(X.shape[0], -1)

        # FC-7
        h3, h3_cache = l.fc_forward(h2, self.model['W2'], self.model['b2'])
        h3, nl_cache3 = l.relu_forward(h3)

        # Softmax
        score, score_cache = l.fc_forward(h3, self.model['W3'], self.model['b3'])

        return score, (X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3)

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3 = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # FC-7
        dh3, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dh3 = self.backward_nonlin(dh3, nl_cache3)

        dh2, dW2, db2 = l.fc_backward(dh3, h3_cache)
        dh2 = dh2.ravel().reshape(hpool.shape)

        # Pool-1
        dpool = l.maxpool_backward(dh2, hpool_cache)

        # Conv-1
        dh1 = self.backward_nonlin(dpool, nl_cache1)
        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(D * 14 * 14, H) / np.sqrt(D * 14 * 14 / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((D, 1)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C))
        )
# ========== CNN - End ==========
