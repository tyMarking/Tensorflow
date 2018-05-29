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
import random
import networkx as nx

from nodes import Node
from netNodeWrapper import NetNode
from connections import Connection

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
    G=nx.random_geometric_graph(50,0.35)
    pos=nx.get_node_attributes(G,'pos')
    
    dmin=1
    ncenter=0
    
    wrapperNodes = []
    
    
    for n in pos:
        node = Node([])
        netNode = NetNode(node, G.node[n])
        wrapperNodes.append(netNode)
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d

    p=nx.single_source_shortest_path_length(G,ncenter)
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        netNodeX = G.node[edge[0]]
        netNodeY = G.node[edge[1]]
        nodeX, nodeY = None, None
        for n in wrapperNodes:
            if n.netNode == netNodeX:
                nodeX = n
            if n.netNode == netNodeY:
                nodeY = n
        
        #normal distrabution of densities for now 
        nodeX.node.addConnection(Connection(nodeY.node, random.random()))
        
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        
        
        
    
#    
#    for node, adjacencies in enumerate(G.adjacency_list()):
#        node_trace['marker']['color'].append(len(adjacencies))
#        node_info = '# of connections: '+str(len(adjacencies))
#        node_trace['text'].append(node_info)
#        
        
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plotly.offline.plot(fig, filename='networkx'    )
main()
