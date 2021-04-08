#!/usr/bin/env python
# coding: utf-8

class map():
    def __init__(self, data):
        self.map_edges = []
        self.final_edges = []
        self.setEdges(data)
        
    def setEdges(self, data):
        for i in range(1,len(data)):
            self.map_edges.append(data[i])
        
    def getEdges(self):
        return self.map_edges
        

