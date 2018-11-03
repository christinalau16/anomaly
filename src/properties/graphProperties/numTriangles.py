import networkx as nx

def numTriangles(G):
    #each triangle is counted three times, once at each node
    num = sum(nx.triangles(G).values())/3
    print("Number of triangles: " + str(num))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
graph = convert(emailSet)
numTriangles(graph)
