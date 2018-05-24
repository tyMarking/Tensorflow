# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:50:11 2018

@author: 136029
"""
from sortedcontainers import SortedSet
from nodes import Node
from connections import Connection
def activate():
        node1 = Node([])
        node1.runActiv = 10
        node2 = Node([])
#        node2.runActiv = 2
        node3 = Node([])
#        node3.runActiv = 3
        node4 = Node([])
        
        node1.addConnection(Connection(node2, 1))
        node1.addConnection(Connection(node3, 0.5))
        node2.addConnection(Connection(node3, 1))
        node3.addConnection(Connection(node4, 1))
        node2.addConnection(Connection(node4, 0.1))
        
        ss = SortedSet([node3,node1,node2,node4], lambda x: -x.runActiv)
        print(ss[0].runActiv)
        print(ss[1].runActiv)
        print(ss[2].runActiv)
        print(ss[3].runActiv)
        
        print("\n\n\n\n\n")
        
        while ss.__len__() > 0:
            #current Node
            cNode = ss.pop(0)
            print(cNode.runActiv)
            cNode.propagate()
#            ss.update()
            ss = SortedSet(ss, lambda x: -x.runActiv)
        print("\n\n\n\n\n")
            
        print(node1.runActiv)
        print(node2.runActiv)
        print(node3.runActiv)
        print(node4.runActiv)
        
#        print(ss[0].runActiv)
#        print(ss[1].runActiv)
#        print(ss[2].runActiv)
        
activate()