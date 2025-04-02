import sys
import argparse
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

import MNIST_util
import nn_layer
import ANN
import CNN
import loss
import optimizer

# ========== Net Trainer - Begin ==========

def get_batch(X, Y, batch_size):
    """Improved batch selection with proper bounds checking"""
    N = len(X)
    max_start = N - batch_size
    if max_start <= 0:
        raise ValueError(f"Batch size {batch_size} exceeds dataset size {N}")
    i = random.randint(0, max_start)
    return X[i:i+batch_size], Y[i:i+batch_size]

def MakeOneHot(Y, D_out):
    """Vectorized one-hot encoding"""
    return np.eye(D_out)[Y]

class MNIST_Trainer:
    def __init__(self, X_train, Y_train, Net='LeNet5', opti='SGDMomentum', H1=300, H2=100):
        # Prepare Data
        self.X_train = X_train
        self.Y_train = Y_train
        self.batch_size = 64
        self.D_out = 10

        print(f"\n{'='*30}")
        print(f"Training {Net} with {opti}")
        if Net == 'ThreeLayerNet':
            print(f"Hidden layers: {H1} → {H2}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*30}")

        # Model initialization
        if Net == 'TwoLayerNet':
            self.model = ANN.TwoLayerNet(784, 400, 10)
        elif Net == 'ThreeLayerNet':
            self.model = ANN.ThreeLayerNet(784, H1, H2, 10)
        elif Net == 'LeNet5':
            self.model = CNN.LeNet5()
            if X_train.ndim != 4:
                self.X_train = self.X_train.reshape(-1, 1, 28, 28)

        # Optimizer configuration
        opti_params = {
            'lr': 0.0001,
            'reg': 0.0001,
            'clip_grad': 1.0
        }
        
        if opti == 'SGD':
            self.opti = optimizer.SGD(self.model.get_params(), **opti_params)
        else:
            opti_params['momentum'] = 0.9
            self.opti = optimizer.SGDMomentum(self.model.get_params(), **opti_params)

        self.criterion = loss.CrossEntropyLoss()
        self.losses = []

    def Train(self, Iter=25000):
        """Enhanced training loop with progress tracking"""
        for i in range(Iter):
            X_batch, Y_batch = get_batch(self.X_train, self.Y_train, self.batch_size)
            Y_onehot = MakeOneHot(Y_batch, self.D_out)
            
            Y_pred = self.model.forward(X_batch)
            loss_val, dout = self.criterion.get(Y_pred, Y_onehot)
            self.model.backward(dout)
            self.opti.step()

            if i % 100 == 0:
                self.losses.append(loss_val)
                print(f"Iter {i:5d}/{Iter} ({i/Iter:.1%}) | Loss: {loss_val:.4f}")

        return self.model

# ========== Net Trainer - End ==========

# ========== Evaluation - Begin ==========

def calculate_accuracy(model, X, Y):
    """Vectorized accuracy calculation"""
    Y_pred = model.forward(X)
    correct = np.sum(np.argmax(Y_pred, 1) == Y)
    acc = correct / X.shape[0]
    label = 'TRAIN' if X.shape[0] == 60000 else 'TEST'
    print(f"{label}--> Correct: {correct}/{X.shape[0]} | Acc: {acc:.4f}")
    return acc

# ========== Evaluation - End ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Neural Network Trainer')
    parser.add_argument('-model', choices=['TwoLayerNet', 'ThreeLayerNet', 'LeNet5'], 
                      required=True, help="Network architecture")
    parser.add_argument('-iter', type=int, default=25000, help="Training iterations")
    parser.add_argument('-opti', choices=['SGDMomentum', 'SGD'], 
                      default='SGDMomentum', help="Optimization algorithm")
    parser.add_argument('-H1', type=int, default=300, 
                      help="First hidden layer size (ThreeLayerNet only)")
    parser.add_argument('-H2', type=int, default=100, 
                      help="Second hidden layer size (ThreeLayerNet only)")
    args = parser.parse_args()

    # Data preparation
    X_train, Y_train, X_test, Y_test = MNIST_util.MNIST_preparation()

    # Architecture-specific preprocessing
    if args.model == 'LeNet5':
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    # Training and evaluation
    trainer = MNIST_Trainer(X_train, Y_train, args.model, args.opti, args.H1, args.H2)
    model = trainer.Train(args.iter)

    # Plot training loss
    plt.plot(np.arange(len(trainer.losses))*100, trainer.losses)
    plt.title(f'Training Loss ({args.model}, {args.opti})')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'loss_{args.model}_{args.opti}.png')
    plt.close()

    # Calculate final accuracies
    calculate_accuracy(model, X_train, Y_train)
    calculate_accuracy(model, X_test, Y_test)