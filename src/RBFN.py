#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import norm, pinv
from loadData import Data

np.random.seed(20)

class RBF:
    def __init__(self, input_dim, num_centers, out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim
        self.beta = [np.random.uniform(-1,1, out_dim) for i in range(num_centers)]
        self.centers = [np.random.uniform(-1,1, input_dim) for i in range(num_centers)]
        self.W = np.random.random((self.num_centers, self.out_dim))
    
    def basicFunc(self, c, d, beta):
        return np.exp(beta * norm(c - d) ** 2)
    
    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.num_centers), dtype = np.float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.basicFunc(c, x, -1/2*(self.beta[ci]**2))
        return G
    
    def train(self, X, Y):
        rnd_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]
        G = self._calcAct(X)
        self.W = np.dot(pinv(G), Y)
        
    def predict(self, X, W):
        G = self._calcAct(X)
        Y = np.dot(G, W)
        return Y
    
    def get_parameter(self):
        return self.input_dim, self.W, self.centers, self.beta
    
    def set_parameter(self, w, centers, beta):
        self.W = w
        self.centers = centers
        self.beta = beta
        
    def get_weight(self):
        return self.W




