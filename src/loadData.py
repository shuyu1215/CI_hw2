#!/usr/bin/env python
# coding: utf-8

from os import listdir,system
from os.path import isfile, join
import numpy as np
from sklearn import preprocessing

class Data():
    def __init__(self):
        self.load = []
        self.load_data()
        self.load4D = []
        self.load4D_Y = []
        self.load_train4D()
        self.load6D = []
        self.load6D_Y = []
        self.load_train6D()
        
    def load_data(self):
        with open('case01.txt','r') as f :
            self.load.clear()
            for line in f.readlines():
                self.load.append(list(map(float,line.strip().split(','))))
            self.load = np.array(self.load)
        origin_point = self.load[0]
    
    def load_train4D(self): #preprocessing data of train4dAll.txt
        temp = []
        with open('data/train4dAll.txt','r') as f :
            self.load4D.clear()
            for line in f.readlines():
                temp = list(map(float,line.strip().split(' ')))
                self.load4D.append(temp[0:3])
                self.load4D_Y.append([temp[3]])
            self.load4D = np.array(self.load4D)
            self.load4D_Y = np.array(self.load4D_Y)
            
    def load_train6D(self): #preprocessing data of train6dAll.txt
        temp = []
        with open('data/train6dAll.txt','r') as f :
            self.load6D.clear()
            for line in f.readlines():
                temp = list(map(float,line.strip().split(' ')))
                self.load6D.append(temp[0:5])
                self.load6D_Y.append([temp[5]])
            self.load6D = np.array(self.load6D)
            self.load6D_Y = np.array(self.load6D_Y)
            
    def load_parameters(self):
        temp = []
        temp_W = []
        temp_centers = []
        temp_beta = []
        with open('data/RBFN_params.txt','r') as f :
            for line in f.readlines():
                temp = list(map(float,line.strip().split(' ')))
                temp_W.append([temp[0]])
                temp_centers.append(temp[1:4])
                temp_beta.append([temp[4]])
            temp_W = np.array(temp_W)
            temp_centers = np.array(temp_centers)
            temp_beta = np.array(temp_beta)
        return temp_W, temp_centers, temp_beta
    
    def getData(self):
        return self.load
    
    def getTrainData4d(self):
        return self.load4D, self.load4D_Y
    
    def getTrainData6d(self):
        return self.load6D, self.load6D_Y
    
    def normalize(self, data):
        self.Min_Max_Scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        MinMax_Data = self.Min_Max_Scaler.fit_transform(data)
        return MinMax_Data
    
    def inverse_normalize(self, data):
        inverse_Data = self.Min_Max_Scaler.inverse_transform(data)
        return inverse_Data
    
    def normalize_Y(self, data):
        self.Scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        MinMax_Data = self.Scaler.fit_transform(data)
        return MinMax_Data
    
    def inverse_normalize_Y(self, data):
        inverse_Data = self.Scaler.inverse_transform(data)
        return inverse_Data
    
    def normalize_input(self, data):
        data_normalized = preprocessing.normalize(data, norm='l2')
        return data_normalized



