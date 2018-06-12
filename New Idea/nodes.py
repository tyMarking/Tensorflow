# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:59:39 2018
Node has:
    connections to other nodes
    ability to grow those connections
    connections are directional (for now)
    connections have node, then weight (percent)
    
    outer reader will connect to these nodes
    
    acive nodes (neuron) and transfer nodes (nerves?)
@author: 136029
"""
import tensorflow as tf
import numpy as np



class Node:
    hasRun = False
    runActiv = 0.0
    connections = []
    numActivates = 0
    
    def __init__(self, connections):
        self.connections = connections
        self.numActicates = 0
    
    def addConnection(self, connection):
        self.connections.append(connection)
        
    def recieveInput(self, strength):
        self.runActiv += strength
        
    def newRun(self):
        self.hasRun = False
        self.runActiv = 0.0
        self.numActivates = 0
        
    def propagate(self):
        for connection in self.connections:
            connection.run(self.runActiv)
        self.hasRun = True
        self.numActivates += 1
        
    def getActivation(self):
        return self.runActiv
    
    