#Generate a graph using the planted partition model
#5 communities, containing 1000, 500, 250, 100, and 50 nodes

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate():
    n = 1900 #number of vertices
    community_labels = initializeCommunities(n)
    p = 0.5
    q = 0.001
    P = [[p, q, q, q, q], [q, p, q, q, q], [q, q, p, q, q], [q, q, q, p, q], [q, q, q, q, p]]

    G = nx.empty_graph(n)
    for i in range(n):
        for j in range(n):
            if i < j:
                communityI = community_labels[i]
                communityJ = community_labels[j]
                val = P[communityI][communityJ]
                p = np.random.rand()
                if p <= val:
                    G.add_edge(i, j)
    return G

#make all parameters
def initializeCommunities(n):
    k = 5 #number of communities
    k_n1 = 1000 #number of vertices in each community
    k_n2 = 500
    k_n3 = 250
    k_n4 = 100
    k_n5 = 50
    c1 = [] #communities
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    communities = [c1, c2, c3, c4, c5]
    z = []

    for i in range(n):
        if i in range(k_n1):
            c1.append(i)
            z.append(0)
        elif i in range(k_n1, k_n1+k_n2):
            c2.append(i)
            z.append(1)
        elif i in range(k_n1+k_n2, k_n1+k_n2+k_n3):
            c3.append(i)
            z.append(2)
        elif i in range(k_n1+k_n2+k_n3, k_n1+k_n2+k_n3+k_n4):
            c4.append(i)
            z.append(3)
        else:
            c5.append(i)
            z.append(4)
    return z

def generateWithNX():
    G = nx.planted_partition_graph(5, 50, 0.5, 0.001)
    return G

generate()
