# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:18:13 2018
density = weight but percent
@author: 136029
"""



class Connection:
    toNode = None
    density = 0.0
    def __init__(self, toNode, density):
        self.toNode = toNode
        self.density = density
        
    def run(self, inputPower):
        self.toNode.recieveInput(inputPower * self.density)
    
    