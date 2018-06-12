# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:50:11 2018

@author: 136029
"""
from sortedcontainers import SortedSet
from nodes import Node
from graphs import visGraph
from connections import Connection
def activate(G):

        nodes = []
        print(G.nodes.items())
        for i in G:
            nodes.append(G.nodes[i]['node'])
        
        print(nodes)
        
        ss = SortedSet(nodes, lambda x: -x.runActiv/(x.numActivates+1))
        
        print("\n\n\n\n\n")
        
        for i in range(10):
            #current Node
            cNode = ss[0]
            print(cNode.runActiv)
            cNode.propagate()
#            ss.update()
            ss = SortedSet(ss, lambda x: -x.runActiv/(x.numActivates+1))
            visGraph(G, i)
        print("spacer")

