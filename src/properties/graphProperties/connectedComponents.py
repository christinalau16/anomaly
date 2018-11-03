import networkx as nx

def sizeConnectedComponents(G):
    num = nx.number_connected_components(G)
    print("number of connected components: " + str(num))
    lenEachCC = [len(c) for c in nx.connected_components(G)]
    print("list of their sizes: " + str(lenEachCC))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
sizeConnectedComponents(convert(emailSet))
