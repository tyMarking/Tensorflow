# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:51:11 2018

@author: 136029
"""

import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt
import random
import networkx as nx

from nodes import Node
from netNodeWrapper import NetNode
from connections import Connection
import graphs

"""
many webs
start with a web


Activations:
    input transition activates certain neurons
    stack of node by highest activation divided by number of previace propigations
    propigates highest node
    untill when?
Density Adjustment
    connection attempts to already tun nodes increase the density?  
    not a training based asjustment
    connections strengthened by activation after activations done 



"""



def main():
#    G=nx.random_lobster(50, 10, 3.4)
    G=nx.soft_random_geometric_graph(500,0.1)
    graphs.visGraph(G)
    
    
main()
