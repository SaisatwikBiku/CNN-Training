from __future__ import print_function
import numpy as np

class SGD:
    def __init__(self, params, lr=0.001, reg=0.0001, clip_grad=1.0):
        """
        SGD with Gradient Clipping and L2 Regularization
        - clip_grad: Maximum absolute gradient value (default: 1.0)
        """
        self.params = params
        self.lr = lr
        self.reg = reg
        self.clip_grad = clip_grad

    def step(self):
        for param in self.params:
            # Gradient clipping
            grad = np.clip(param['grad'], -self.clip_grad, self.clip_grad)
            # Update with L2 regularization
            param['val'] -= self.lr * (grad + self.reg * param['val'])

class SGDMomentum:
    def __init__(self, params, lr=0.001, momentum=0.9, reg=0.0001, clip_grad=1.0):
        """
        SGD with Momentum (correct implementation)
        - momentum: 0.9 is better than 0.99 for deep networks
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum  # Renamed from 'rho' for clarity
        self.reg = reg
        self.clip_grad = clip_grad
        self.velocities = [np.zeros_like(p['val']) for p in params]

    def step(self):
        for i, param in enumerate(self.params):
            # Gradient clipping
            grad = np.clip(param['grad'], -self.clip_grad, self.clip_grad)
            
            # Update velocity (correct momentum formula)
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            
            # Apply update with L2 regularization
            param['val'] -= self.lr * (self.velocities[i] + self.reg * param['val'])