import networkx as nx

def testNode(G, n):
    #degree with weight
    print("degree of " + str(n) + ": ", G.degree(n, weight = 'weight'))

from src.generate.generatePPM import generate
testNode(generate(), 10)
