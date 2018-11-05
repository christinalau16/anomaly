import networkx as nx

def numTriangles(G, n):
    EG = nx.ego_graph(G, n)
    #each triangle is counted three times, once at each node
    numTri = sum(nx.triangles(EG).values())/3
    print("number of triangles in ego-network: " + str(numTri))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
numTriangles(convert(emailSet), 212)
