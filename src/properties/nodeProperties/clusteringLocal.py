import networkx as nx

def localClustering(G, n):
    c = nx.clustering(G, n)
    print("local clustering coefficient: " + str(c))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
localClustering(convert(emailSet), 212)
