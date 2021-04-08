#!/usr/bin/env python
# coding: utf-8

from tkinter import *
import tkinter as tk
from os import listdir,system
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math
import timeit
from loadData import Data
from Car import car
from Map import map
from loadData import Data
from RBFN import RBF
from Genetic import genetic
from scipy.linalg import norm, pinv

class gui(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.windows = master
        self.grid()
        self.create_windows()
        self.data = []
        self.edges = []
        self.file_name = ''
    
    def get_list(self,event):
        self.index = self.listbox.curselection()[0]
        self.selected = self.listbox.get(self.index)
        self.file_name = 'data/'+self.selected

    def create_windows(self):
        self.windows.title("HW2")
        #make list
        mypath = './data/'
        self.files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.listbox = tk.Listbox(windows, width=20, height=3)
        self.listbox.grid(row=0, column=0,columnspan=2,stick=tk.W+tk.E)
        self.yscroll = tk.Scrollbar(command=self.listbox.yview, orient=tk.VERTICAL)
        self.yscroll.grid(row=0, column=2, sticky=tk.W+tk.E)
        self.listbox.configure(yscrollcommand=self.yscroll.set)
        for item in self.files:
            self.listbox.insert(tk.END, item)
        self.listbox.bind('<ButtonRelease-1>', self.get_list)
    
        self.result_figure = Figure(figsize=(8,8), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, self.windows)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky=tk.W+tk.E)
        #region of inputtin parameters
        self.iterations = tk.Label(windows, text="iterations:").grid(row=2,column=0, sticky=tk.W+tk.E)
        self.population = tk.Label(windows, text="population:").grid(row=3,column=0, sticky=tk.W+tk.E)
        self.mutation_rate = tk.Label(windows, text="mutation rate:").grid(row=4,column=0, sticky=tk.W+tk.E)
        self.crossover_rate = tk.Label(windows, text="cross rate:").grid(row=5,column=0, sticky=tk.W+tk.E)
        self.e1 = tk.Entry(windows)
        self.e2 = tk.Entry(windows)
        self.e3 = tk.Entry(windows)
        self.e4 = tk.Entry(windows)
        self.e1.grid(row=2, column=1, sticky=tk.W+tk.E)
        self.e2.grid(row=3, column=1, sticky=tk.W+tk.E)
        self.e3.grid(row=4, column=1, sticky=tk.W+tk.E)
        self.e4.grid(row=5, column=1, sticky=tk.W+tk.E)
        self.e1.delete(0,'end')
        self.e2.delete(0,'end')
        self.e3.delete(0,'end')
        self.e4.delete(0,'end')
        self.e1.insert(10,50)
        self.e2.insert(10,50)
        self.e3.insert(10,0.2)
        self.e4.insert(10,0.8)
        self.show = tk.Button(self.windows, text='Quit', command=windows.quit).grid(row=1, column=0, sticky=tk.W+tk.E)
        self.show = tk.Button(self.windows, text='Show', command=self.run).grid(row=1, column=1, sticky=tk.W+tk.E)
    
    def run(self):
        data4D = open("data/RBFN_params.txt",'w+')
        d = Data() #new object of selected data
        self.data = d.getData()
        if(self.file_name == 'data/train4dAll.txt'):
            x, y = d.getTrainData4d()
        elif(self.file_name == 'data/train6dAll.txt'):
            x, y = d.getTrainData6d()
        else:
            x, y = d.getTrainData4d()
        #normalize training data
        x = d.normalize_input(x)
        y = d.normalize_Y(y)
        #get input parameters
        cross_rate = float(self.e4.get())
        mutation_rate = float(self.e3.get())
        pop_size = int(self.e2.get())
        iterations = int(self.e1.get())
        if(self.file_name != 'data/RBFN_params.txt'):
            #training weight --start--
            rbf = RBF(3, 50, 1) #new object of RBFN
            x_dim, w, centers, beta = rbf.get_parameter()
            ga = genetic(len(x), cross_rate, mutation_rate, pop_size, x_dim, x, y, w, centers, beta) #new object of ga
            temp = 100
            for i in range(0, iterations):
                fitness = ga.F()
                best_idx = np.argmin(fitness)
                if(fitness[best_idx] < temp):
                    temp = fitness[best_idx]
                    best_DNA = ga.get_best_DNA(best_idx)
                    ga.update(best_DNA)
            #training weight --end--
            ga.w, ga.centers, ga.beta = ga.get_data()
            rbf.set_parameter(ga.w, ga.centers, ga.beta)
            rbf.train(x,y)
            z1 = rbf.predict(x, rbf.get_weight())
            y = d.inverse_normalize_Y(y)
            z = d.inverse_normalize_Y(z1)
            dim, output_W, output_centers, output_beta = rbf.get_parameter()
            for i in range(0, len(output_W)): #save trained parameters
                print(str(output_W[i][0]), str(output_centers[i][0]),str(output_centers[i][1]),str(output_centers[i][2]), str(output_beta[i][0]),file=data4D)
            data4D.close()
        else:
            load_w, load_centers, load_beta = d.load_parameters()
            rbf = RBF(3, len(load_centers), 1)
            rbf.set_parameter(load_w, load_centers, load_beta)
        m = map(self.data) #new object of map
        self.edges = m.getEdges()
        c = car(self.data[0]) #new object of car
        Y = c.getPosition() #get the position of car
        self.mapping(self.edges)
        while(Y[1] < 43):
            self.draw_car(c.getPosition()) #draw car on the window
            c.sensor(self.edges) #calculate the distane of front, left, right
            F, R, L = c.getDistance()
            input_x = []
            if(self.file_name == 'data/train4dAll.txt'):
                input_x.append([F, R, L])
            elif(self.file_name == 'data/train6dAll.txt'):
                pos = c.getPosition()
                input_x.append([pos[0], pos[1], F, R, L])
            else:
                input_x.append([F, R, L])
            input_x = np.array(input_x)
            input_x = d.normalize_input(input_x)
            wheel = rbf.predict(input_x, rbf.get_weight()) #predict the degree of wheel
            wheel = d.inverse_normalize_Y(wheel)
            if wheel < -40:
                wheel = -40
            elif wheel > 40:
                wheel = 40
            c.setWheel(wheel)
            c.update_car_direction()
            c.update_car_pos()
            
            
    def mapping(self, edges):
        self.result_figure.clf()
        self.draw_map(edges[0],edges[1], 'r')
        self.draw_map([-6,0], [6, 0], 'black')
        for i in range(2,len(edges) - 1):
            self.draw_map(edges[i],edges[i+1], 'b')
    
    def draw_map(self,p_1, p_2, c):
        self.result_figure.a = self.result_figure.add_subplot(111)
        min_v, max_v, ymin, ymax = self.set_maxmin(p_1, p_2)
        x = np.linspace(min_v, max_v)
        if (p_2[0] - p_1[0] == 0):
            self.result_figure.a.vlines(p_1[0], ymin, ymax, color = c)
        else:
            slope = (p_2[1] - p_1[1])/(p_2[0] - p_1[0])
            intercept = p_1[1]
            y = (slope*x) + intercept
            self.result_figure.a.plot(x, y, color = c)
        self.result_figure.a.set_title('Map')
        self.result_canvas.draw()
        
    def set_maxmin(self, p1, p2):
        if p1[0] < p2[0]:
            x_m = p1[0]
            x_M = p2[0]
        else:
            x_m = p2[0]
            x_M = p1[0]  
        if p1[1] < p2[1]:
            y_m = p1[1]
            y_M = p2[1]
        else:
            y_m = p2[1]
            y_M = p1[1]
        return x_m, x_M, y_m, y_M
    
    def draw_car(self,point):
        self.result_figure.b = self.result_figure.add_subplot(111)
        p_x = point[0]
        p_y = point[1]
        r = 3.0
        theta = np.arange(0, 2*np.pi, 0.01)
        x = p_x + r * np.cos(theta)
        y = p_y + r * np.sin(theta)
        self.result_figure.b.plot(p_x, p_y, 'ro')
        self.result_figure.b.plot(x, y, color = 'g')
        self.result_canvas.draw()


if __name__ == "__main__":
    windows = tk.Tk()
    app = gui(windows)
    windows.mainloop()




