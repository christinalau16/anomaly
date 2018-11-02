#Still need to put in logarithmic scale
import collections
import matplotlib.pyplot as plt
import networkx as nx

def degreeGraphHistogram(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

from src.generate.generatePPM import generate
degreeGraphHistogram(generate())
