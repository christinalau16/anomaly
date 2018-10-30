#Still need to put in logarithmic scale
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def outDegree():
    G = nx.scale_free_graph(10)
    print([d for n, d in G.out_degree()])

    degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)  # degree sequence
    print ("Degree sequence", degree_sequence)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Out Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

def inDegree():
    G = nx.scale_free_graph(10)
    print([d for n, d in G.in_degree()])

    degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)  # degree sequence
    print ("Degree sequence", degree_sequence)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("In Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

outDegree()
inDegree()
