import networkx as nx
#import snap

def diameter(G):
    diameter = nx.diameter(G)
    print("diameter: " + str(diameter))

#def effectiveDiameter(G):
    #effDiam = snap.GetBfsEffDiam(G, 10, false)
    #print("90-percentile effective diameter: " + str(effDiam))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
diameter(convert(politicalSet))
#effectiveDiameter(graph)
