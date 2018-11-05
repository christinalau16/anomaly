import networkx as nx

def numEdges(G, n):
    EG = nx.ego_graph(G, n)
    numEdges =  EG.number_of_edges()
    print("number of edges in ego-network: " + str(numEdges))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
numEdges(convert(emailSet), 212)
