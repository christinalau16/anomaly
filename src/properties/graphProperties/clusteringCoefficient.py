import networkx as nx

def clusteringCoefficient(G):
    average = nx.average_clustering(G)
    print("global clustering coefficient: " + str(average))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
graph = convert(politicalSet)
clusteringCoefficient(graph)
