#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import importer
from loadData import Data
from RBFN import RBF
from scipy.linalg import norm, pinv

class genetic:
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, x_dim, x, y, w, centers, beta):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.len_w = len(w)
        self.w = w
        self.centers = centers
        self.num_centers = len(centers) * x_dim
        self.beta = beta
        self.len_beta = len(beta) * x_dim
        self.x = x
        self.x_dim = x_dim
        self.y = y
        self.theta = np.random.uniform(-1,1)
        self.DNA = self.getDNA(self.theta, w, centers, beta)
        self.pop = np.vstack(self.DNA for _ in range(pop_size))
        #self.print_data()
        
    
    def print_data(self):
        print('self.x:', self.x)
        print('self.w', self.w)
        print('self.centers: ',self.centers)
        print('self.beta:', self.beta)
        print('self.DNA:', self.DNA)
        print('self.pop:', self.pop)
    
    def getDNA(self, theta, w, centers, beta):
        genetic = []
        print('-------DNA--------')
        print(theta)
        print(w)
        print(centers)
        print(beta)
        genetic = np.hstack((genetic,theta))
        for k in range(0, len(w)):
            genetic = np.hstack((genetic, w[k]))
        for i in range(0, len(centers)):
            genetic = np.hstack((genetic, centers[i]))
        for j in range(0, len(beta)):
            genetic = np.hstack((genetic, beta[j]))
        return genetic

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]


    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep = parent[~cross_points]                                       # find the crossover number
            swap = pop[i_, np.isin(pop[i_].ravel(), keep, invert=True)]
            parent[:] = np.concatenate((keep, swap))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child
    
    def getDim(self, input_num, input_dim):
        Dim = 1 + input_num + input_num * input_dim + input_num
        return Dim
    
    def basicFunc(self, c, d, beta):
        return np.exp(beta * norm(c - d) ** 2)
    
    def F(self):  #compute the value of function
        fitness = []
        for i in range(0,len(self.pop)):
            self.update(self.pop[i])
            rnd_idx = np.random.permutation(self.x.shape[0])[:len(self.centers)]
            self.centers = [self.x[i, :] for i in rnd_idx]
            G = self._calcAct()
            self.w = np.dot(pinv(G), self.y)
            f = np.dot(G, self.w)
            fitness.append(self.E(self.y,f))
        return fitness
    
    def _calcAct(self):
        G = np.zeros((self.x.shape[0], len(self.centers)), dtype = np.float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(self.x):
                G[xi, ci] = self.basicFunc(c, x, -1/2*(self.beta[ci]**2))
        return G
    
    def get_fitness(self,pred):
        return pred + 1e-3 - np.min(pred)
    
    def E(self, y, f): #compute fitness value
        total = 0
        for i in range(0, len(y)):
            total += (y[i] - f[i])**2
        total = total/2
        return total
    
    def evolve(self, fitness): #evolution function
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
        
    def update(self, DNA): #update the best DNA
        self.theta = DNA[0]
        self.w = DNA[1:len(self.w)+1]
        temp_w = []
        for i in range(0, len(self.w)):
            temp_w.append([self.w[i]])
        self.w = temp_w
        self.centers = DNA[1+len(self.w):(1+len(self.w)+self.num_centers)]
        self.centers = np.split(self.centers, self.num_centers/self.x_dim, axis=0)
        self.beta = DNA[1+len(self.w)+self.num_centers:]
        self.beta = np.split(self.beta, self.len_beta/self.x_dim, axis=0)
    
    def get_best_DNA(self, idx):
        return self.pop[idx]
    
    def get_weight(self):
        return self.w
    
    def get_data(self):
        return self.w, self.centers, self.beta
    
    def set_data(self, w, centers, beta):
        self.w = w
        self.centers = centers
        self.beta = beta
    
    def get_theta(self):
        return self.theta
