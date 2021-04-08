#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

class car():
    def __init__(self, pos):
        self.position = []
        self.direction = 0.0
        self.wheel = 0.0
        self.dist_F = 0.0
        self.dist_L = 0.0
        self.dist_R = 0.0
        self.circle_F = []
        self.circle_L = []
        self.circle_R = []
        for i in pos:
            self.position.append(i)
        self.direction = self.position[2]
        
    def getPosition(self):
        return self.position
    
    def setPosition(self,x, y):
        self.position[0] = x
        self.position[1] = y
    
    def getDirection(self):
        return self.direction
    
    def setDirection(self,d):
        self.direction = d
        
    def getWheel(self):
        return self.wheel
    
    def setWheel(self, w):
        self.wheel = w
    
    def getDistance(self):
        return self.dist_F, self.dist_R, self.dist_L
        
    def sensor(self, edges): #calculate distanes
        self.dist_F = 1000.0
        self.dist_L = 1000.0
        self.dist_R = 1000.0
        self.setCirclePoint(self.position)
        for i in range(2,len(edges) - 2):
            temp_F = self.GetIntersectPoint(self.circle_F,edges[i], edges[i+1])
            if(temp_F < self.dist_F):
                self.dist_F = round(temp_F)
        for j in range(2,len(edges) - 2):
            temp_L = self.GetIntersectPoint(self.circle_L,edges[j], edges[j+1])
            if(temp_L < self.dist_L):
                self.dist_L = round(temp_L)
        for k in range(2,len(edges) - 2):
            temp_R = self.GetIntersectPoint(self.circle_R,edges[k], edges[k+1])
            if(temp_R < self.dist_R):
                self.dist_R = round(temp_R)
        
    def setCirclePoint(self, p):
        self.resetCircle()
        cos_F, sin_F = self.setCosSin(0)
        cos_R, sin_R = self.setCosSin(-45)
        cos_L, sin_L = self.setCosSin(45)
        self.circle_F.append(round(p[0] + cos_F * 3))
        self.circle_F.append(round(p[1] + sin_F * 3))
        self.circle_R.append(round(p[0] + cos_R * 3))
        self.circle_R.append(round(p[1] + sin_R * 3))
        self.circle_L.append(round(p[0] + cos_L * 3))
        self.circle_L.append(round(p[1] + sin_L * 3))
    
    def resetCircle(self):
        self.circle_F.clear()
        self.circle_L.clear()
        self.circle_R.clear()
    
    def caculate_D(self,circle_p, edges1, edges2):
        x1 = self.position[0]
        y1 = self.position[1]
        x2 = circle_p[0]
        y2 = circle_p[0]

        x3 = edges1[0]
        y3 = edges1[1]
        x4 = edges2[0]
        y4 = edges2[1]
        
        if(x2 - x1) == 0:
            k1 = None
            b1 = 0
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)
            b1 = y1 * 1.0 - x1 * k1 * 1.0
            
        if(x4 - x3) == 0:
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            b2 = y3 * 1.0 - x3 * k2 * 1.0
            
        if k2 == None:
            x = x3
        elif k1 == None:
            x = x1
        elif k1 == None and k2 == None:
            return 50.0
        elif k1 - k2 == 0:
            return 50.0
        else:
            x = (b2 - b1)*1.0/(k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        if (x >= min(x3, x4)-1 and x <= max(x3, x4)+1) and (y >= min(y3, y4)-1 and y <= max(y3, y4)+1):
            Distance = self.dist(x, y) 
            return Distance
        else:
            return 50.0
                             
    def dist(self, cross_x, cross_y):
        D = np.sqrt((cross_x - self.position[0])**2 + (cross_y - self.position[1])**2)
        return D
    
    def update_car_direction(self):
        if self.wheel == 0:
            arcsin_t = 0
        else:
            sin_W = np.sin(math.radians(self.wheel))
            arcsin_t = np.degrees(np.arcsin((2 * sin_W) / 6))
        self.direction = self.direction - arcsin_t
        
    def update_car_pos(self):
        now_X, now_Y= self.position[0], self.position[1]
        car_d_m = math.radians(self.direction)
        wheel_d_m = math.radians(self.wheel)
        sin_car = np.sin(car_d_m)
        cos_car = np.cos(car_d_m)
        sin_wheel = np.sin(wheel_d_m)
        cos_wheel = np.cos(wheel_d_m)
        sin_c_w = np.sin((car_d_m + wheel_d_m))
        cos_c_w = np.cos((car_d_m + wheel_d_m))
        if self.direction == 90:
            cos_car = 0
        elif self.direction == 0:
            sin_car = 0
        elif self.wheel == 90:
            cos_wheel = 0
        elif self.wheel == 0:
            sin_wheel = 0
        if self.direction + self.wheel == 90:
            cos_c_w = 0
        elif (self.direction + self.wheel) == 0:
            sin_c_w = 0
        self.position[0] = now_X + cos_c_w + (sin_wheel * sin_car)
        self.position[1] = now_Y + sin_c_w - (sin_wheel * cos_car)
        
    def setCosSin(self, diff):
        COS = np.cos(math.radians(self.direction + diff))
        SIN = np.sin(math.radians(self.direction + diff))
        if (self.direction + diff) == 90:
            return 0, SIN
        elif (self.direction + diff) == 0:
            return COS, 0
        else:
            return COS, SIN
        
    def GeneralEquation(self, first_x,first_y,second_x,second_y):
        A=second_y-first_y
        B=first_x-second_x
        C=second_x*first_y-first_x*second_y
        return A,B,C
    
    def GetIntersectPoint(self,circle_p, edges1, edges2):
        x1 = self.position[0]
        y1 = self.position[1]
        x2 = circle_p[0]
        y2 = circle_p[1]

        x3 = edges1[0]
        y3 = edges1[1]
        x4 = edges2[0]
        y4 = edges2[1]
        
        A1,B1,C1 = self.GeneralEquation(x1,y1,x2,y2)
        A2, B2, C2 = self.GeneralEquation(x3,y3,x4,y4)
        m = A1 * B2 - A2 * B1
        if m == 0:
            print('no intersection')
            return 1000.0
        else:   #if no intersection, return 1000
            x = (C2 * B1 - C1 * B2) / m
            y = (C1 * A2 - C2 * A1) / m
            if x >= min(x3, x4)-1 and x <= max(x3, x4)+1 and y >= min(y3, y4)-1 and y <= max(y3, y4)+1:
                if x2 - x1 > 0:
                    if y2 - y1 > 0:
                        if x >= min(x1, x2) and y >= min(y1, y2):
                            Distance = self.dist(x, y)
                            return Distance
                        else:
                            print('but no intersection')
                            return 1000.0
                    elif y2 - y1 <= 0:
                        if x >= min(x1, x2) and y <= max(y1, y2):
                            Distance = self.dist(x, y)
                            return Distance
                        else:
                            print('but no intersection')
                            return 1000.0
                else:
                    if y2 - y1 > 0:
                        if x <= max(x1, x2) and y >= min(y1, y2):
                            Distance = self.dist(x, y)
                            return Distance
                        else:
                            print('but no intersection')
                            return 1000.0
                    elif y2 - y1 <= 0:
                        if x <= max(x1, x2) and y <= max(y1, y2):
                            Distance = self.dist(x, y)
                            return Distance
                        else:
                            print('but no intersection')
                            return 1000.0
            else:
                print('but no intersection')
                return 1000.0


