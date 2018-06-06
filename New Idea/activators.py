# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:50:11 2018

@author: 136029
"""
from sortedcontainers import SortedSet
from nodes import Node
from connections import Connection
def activate():

        
        ss = SortedSet([node3,node1,node2,node4], lambda x: -x.runActiv)
        
        print("\n\n\n\n\n")
        
        while ss.__len__() > 0:
            #current Node
            cNode = ss.pop(0)
            print(cNode.runActiv)
            cNode.propagate()
#            ss.update()
            ss = SortedSet(ss, lambda x: -x.runActiv)

activate()