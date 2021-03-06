# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:26:24 2018

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
import pandas


from nodes import Node
from netNodeWrapper import NetNode
from connections import Connection

def getGeoGraph():
    G=nx.soft_random_geometric_graph(5000,0.07)

    neurons = {}
    for i in range(len(G.nodes)):
         x, y = G.node[i]['pos']
         neurons[i] = Node([])
         neurons[i].runActiv = 0
    neurons[250].runActiv = 10
    nx.set_node_attributes(G, neurons, 'node')
    densities = {}
    for edge in G.edges:
        density  = random.gauss(0.5,0.3)
        while density < 0 or density > 1:
            density  = random.gauss(0.5,0.3)
        densities[edge] = density
        neurons[edge[0]].addConnection(Connection(neurons[edge[1]], density))
        neurons[edge[1]].addConnection(Connection(neurons[edge[0]], density))
        
    nx.set_edge_attributes(G, densities, 'density')
    print(G.nodes(data=True))
         
    return G
     
def visGraph(G, index):

    pos=nx.get_node_attributes(G,'pos')
    
    dmin=1
    ncenter=0
    
    
    
    for n in pos:

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
                title='Neuron Activations',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        
        
        
    
    
    for node in G.nodes:
        
        node_trace['marker']['color'].append(G.nodes[node]['node'].getActivation())
        node_info = 'Activation: '+str(G.nodes[node]['node'].getActivation())
        node_trace['text'].append(node_info)
        
        
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

    plotly.offline.plot(fig, filename='networkx_' + str(index) + '.html'    )